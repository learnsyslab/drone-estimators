"""Integration tests of the Kalman filter with crazyflow's dynamics."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from crazyflow.dynamics import available_dynamics, dynamics_features, supported_drones
from scipy.spatial.transform import Rotation as R

from drone_estimators.estimator_kalman import KalmanFilter
from drone_estimators.structs.estimator_data import EstimatorData
from drone_estimators.utils.dynamics import dynamics_function

# All dynamics with every drone they have parameters for
DYNAMICS_DRONES = [(d, drone) for d in available_dynamics for drone in supported_drones(d)]


# TODO test if the whole filterpy chain is jitable


@pytest.mark.integration
@pytest.mark.parametrize("dynamics, drone", DYNAMICS_DRONES)
def test_kalman(dynamics: str, drone: str):
    """Tests if the Kalman filter can be imported and stepped."""
    supports_dynamics = dynamics_features(dynamics_function(dynamics, drone))["rotor_dynamics"]
    kf = KalmanFilter(1 / 200, dynamics, drone, estimate_rotor_vel=supports_dynamics)

    kf.predict(1 / 240, np.array([0.0, 0.0, 0.0, 0.5]))

    kf.correct(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.integration
@pytest.mark.parametrize("jit", [False, True])
def test_kalman_batched(jit: bool):
    """Tests that a batched Kalman filter matches one (numpy) filter per drone.

    The drones have different time steps and not every drone gets a measurement in every step.
    """
    n_drones, n_steps = 3, 40
    kwargs = dict(estimate_rotor_vel=True, estimate_dist_f=True)
    rng = np.random.default_rng(0)
    t = np.arange(n_steps)[:, None] / 200
    pos = rng.normal(size=(n_drones, 1, 3)) + 0.01 * np.sin(5 * t)
    quat = R.from_euler("xyz", 0.05 * np.sin(rng.uniform(5, 10, (n_drones, 1, 3)) * t)).as_quat()
    cmd = np.concat((np.zeros((n_drones, n_steps, 3)), np.full((n_drones, n_steps, 1), 0.3)), -1)
    dt = rng.uniform(1 / 400, 1 / 100, (n_steps, n_drones))
    dt[rng.random((n_steps, n_drones)) < 0.2] = 0.0  # Not predicted, and therefore not corrected
    has_meas = (rng.random((n_steps, n_drones)) < 0.7) & (dt > 0)
    with jax.enable_x64(True):
        args = (1 / 200, "so_rpy_rotor_drag", "cf21B_500")
        batched = KalmanFilter(*args, batch_shape=(n_drones,), jit=jit, **kwargs)
        batched.set_state(pos[:, 0], quat[:, 0])
        single = [KalmanFilter(*args, **kwargs) for _ in range(n_drones)]
        for i, kf in enumerate(single):
            kf.set_state(pos[i, 0], quat[i, 0])
        for k in range(1, n_steps):
            batched.predict(dt[k], cmd[:, k])
            batched.correct(pos[:, k], quat[:, k], mask=has_meas[k])
            for i, kf in enumerate(single):
                kf.predict(dt[k, i], cmd[i, k])
                if has_meas[k, i]:
                    kf.correct(pos[i, k], quat[i, k])
    for i, kf in enumerate(single):
        x, x_single = (
            EstimatorData.as_state_array(batched.data)[i],
            EstimatorData.as_state_array(kf.data),
        )
        P, P_single = batched.data.covariance[i], kf.data.covariance
        # The UKF amplifies floating point differences (e.g. from XLA), so we compare to the scale
        assert np.allclose(x, x_single, rtol=0, atol=1e-5 * np.max(np.abs(x_single)))
        assert np.allclose(P, P_single, rtol=0, atol=1e-5 * np.max(np.abs(P_single)))
