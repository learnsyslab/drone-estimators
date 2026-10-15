"""Unit tests of the legacy estimator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from drone_estimators.estimator_legacy import (
    LegacyData,
    LegacySettings,
    StateEstimator,
    legacy_correct,
)

LEGACY_PARAMS = (0.0001, 0.007, 0.09, 0.005, 0.07)  # As in the ROS node


@pytest.mark.unit
def test_legacy_jit():
    """Tests if the legacy estimator can be jitted and matches the non-jitted version."""
    settings = LegacySettings(*LEGACY_PARAMS)
    dt = 1 / 200
    t = np.arange(1, 101) * dt
    pos = np.stack([np.sin(t), np.cos(t), t], axis=-1)
    quat = R.from_rotvec(np.outer(t, [1.0, 2.0, 3.0])).as_quat()
    quat[::3] *= -1  # Same rotation, other hemisphere
    pos[50], quat[50] = pos[49], quat[49]  # Repeated measurement
    with jax.enable_x64(True):
        correct_jit = jax.jit(legacy_correct)
        data_np = LegacyData.create_empty()
        data_jit = jax.tree.map(jnp.asarray, data_np)
        for k in range(len(t)):
            step_dt = 0.0 if k == 70 else dt  # Correction without time passing
            data_np = legacy_correct(data_np, settings, pos[k], quat[k], step_dt)
            data_jit = correct_jit(data_jit, settings, pos[k], quat[k], np.float64(step_dt))
        for x_np, x_jit in zip(jax.tree.leaves(data_np), jax.tree.leaves(data_jit)):
            assert x_jit.dtype == jnp.float64
            assert np.all(np.isfinite(x_jit))
            assert np.allclose(x_np, x_jit, rtol=1e-9, atol=1e-9)


@pytest.mark.unit
def test_legacy_startup():
    """Tests that the first measurement after initialization does not cause a velocity spike."""
    pos = np.array([0.3, -0.5, 1.0])
    quat = R.from_euler("z", 1.0).as_quat()
    with jax.enable_x64(True):
        estimator = StateEstimator(LEGACY_PARAMS)
        estimator.set_state(pos, quat)
        estimator.predict(1 / 100)
        data = estimator.correct(pos, -quat)  # Same rotation, other hemisphere
    assert np.allclose(data.pos, pos)
    assert np.allclose(np.abs(data.quat @ quat), 1.0)
    assert np.allclose(data.vel, 0.0)
    assert np.allclose(data.ang_vel, 0.0)


@pytest.mark.unit
def test_legacy_quat_sign():
    """Tests that the sign of the measured quaternions does not change the estimate."""
    t = np.arange(100) / 200
    quat = R.from_rotvec(np.outer(t, [1.0, 2.0, 3.0])).as_quat()
    signs = np.where(np.random.default_rng(0).random((100, 1)) < 0.5, -1.0, 1.0)
    estimates = []
    with jax.enable_x64(True):
        for q in (quat, quat * signs):
            estimator = StateEstimator(LEGACY_PARAMS)
            estimator.set_state(np.zeros(3), q[0])
            for k in range(1, 100):
                estimator.predict(1 / 200)
                data = estimator.correct(np.zeros(3), q[k])
                assert np.isclose(np.linalg.norm(data.quat), 1.0)
            estimates.append(data)
    assert np.isclose(np.abs(estimates[0].quat @ estimates[1].quat), 1.0)
    assert np.allclose(estimates[0].ang_vel, estimates[1].ang_vel)


@pytest.mark.unit
def test_legacy_body_rates():
    """Tests that the estimated angular velocity is in body coordinates."""
    ang_vel = np.array([1.0, -2.0, 3.0])
    t = np.arange(400) / 200
    rot0 = R.from_euler("xyz", [0.3, -0.5, 1.0])
    quat = (rot0 * R.from_rotvec(np.outer(t, ang_vel))).as_quat()
    with jax.enable_x64(True):
        estimator = StateEstimator(LEGACY_PARAMS)
        estimator.set_state(np.zeros(3), quat[0])
        for k in range(1, len(t)):
            estimator.predict(1 / 200)
            data = estimator.correct(np.zeros(3), quat[k])
    assert np.allclose(data.ang_vel, ang_vel, atol=1e-6)
    assert np.isclose(np.abs(data.quat @ quat[-1]), 1.0)


@pytest.mark.unit
def test_legacy_needs_x64():
    """Tests that the legacy estimator refuses to run in 32 bit."""
    with jax.enable_x64(False), pytest.raises(RuntimeError):
        StateEstimator(LEGACY_PARAMS)


@pytest.mark.unit
@pytest.mark.parametrize("jit", [False, True])
def test_legacy_batched(jit: bool):
    """Tests that a batched estimator matches one estimator per drone.

    Not every drone gets a measurement in every step, which is handled by the mask.
    """
    n_drones, n_steps, dt = 3, 50, 1 / 200
    rng = np.random.default_rng(0)
    t = np.arange(n_steps) * dt
    pos = rng.normal(size=(n_drones, 1, 3)) + np.stack([np.sin(t), np.cos(t), t], axis=-1)
    quat = R.from_rotvec(rng.normal(size=(n_drones, 1, 3)) * t[:, None]).as_quat()
    has_meas = rng.random((n_steps, n_drones)) < 0.7
    with jax.enable_x64(True):
        batched = StateEstimator(LEGACY_PARAMS, batch_shape=(n_drones,), jit=jit)
        batched.set_state(pos[:, 0], quat[:, 0])
        single = [StateEstimator(LEGACY_PARAMS) for _ in range(n_drones)]
        for i, estimator in enumerate(single):
            estimator.set_state(pos[i, 0], quat[i, 0])
        for k in range(1, n_steps):
            # Invalid measurements of drones without a measurement must not be used
            quat_k = np.where(has_meas[k, :, None], quat[:, k], 0.0)
            batched.predict(dt)
            batched.correct(pos[:, k], quat_k, mask=has_meas[k])
            for i, estimator in enumerate(single):
                estimator.predict(dt)
                if has_meas[k, i]:
                    estimator.correct(pos[i, k], quat[i, k])
    for i, estimator in enumerate(single):
        for x, x_single in zip(jax.tree.leaves(batched.data), jax.tree.leaves(estimator.data)):
            assert np.allclose(x[i], x_single, rtol=1e-9, atol=1e-9)
