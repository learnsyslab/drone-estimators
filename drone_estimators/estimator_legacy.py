"""A simple smoothing estimator.

This estimator is originally from Chris McKinnon.
It was adapted to work with similarly to the new estimators for backwards compatability.

The filter itself is the pure function legacy_correct, which the StateEstimator class jits.
Quaternions are scalar last and all rotations use scipy's Rotation.
"""

from __future__ import absolute_import, annotations, division, print_function

import os

# scipy's Rotation only works with jax arrays (and therefore in jit) if the array API is enabled.
# This has to be set before scipy is imported, as drone_models does as well.
os.environ["SCIPY_ARRAY_API"] = "1"

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from scipy.spatial.transform import Rotation as R

from drone_estimators.structs.estimator_data import EstimatorData

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


@dataclass
class LegacySettings:
    """Time constants of the low pass filters."""

    tau_est_trans: float
    tau_est_trans_dot: float
    tau_est_trans_dot_dot: float
    tau_est_rot: float
    tau_est_rot_dot: float


@dataclass
class LegacyData:
    """State of the legacy estimator."""

    # Estimated states
    pos: Array
    vel: Array
    acc: Array  # Not published, only used for the prediction
    quat: Array
    ang_vel: Array  # (body frame)

    # Numeric derivatives of the measurements
    vel_meas: Array
    acc_meas: Array
    ang_vel_meas: Array

    # Previous measurements
    pos_prev: Array
    vel_prev: Array
    quat_prev: Array

    @classmethod
    def create_empty(cls) -> LegacyData:
        """Create the initial state."""
        zeros = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        return cls(
            pos=zeros,
            vel=zeros,
            acc=zeros,
            quat=quat,
            ang_vel=zeros,
            vel_meas=zeros,
            acc_meas=zeros,
            ang_vel_meas=zeros,
            pos_prev=zeros,
            vel_prev=zeros,
            quat_prev=quat,
        )


class StateEstimator(object):
    """Vicon state estimation and filtering.

    Parameters
    ----------
    filter_parameters : sequence of floats
        The 5 time constants of the low pass filters, see LegacySettings
    """

    def __init__(self, filter_parameters: tuple):
        """TODO."""
        # The numeric derivatives divide by dt (twice for the acceleration). In 32 bit, this
        # amplifies the rounding errors of the measurements too much.
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "The legacy estimator needs 64 bit precision. "
                'Call jax.config.update("jax_enable_x64", True) before creating it.'
            )
        self.settings = LegacySettings(*filter_parameters)
        self.data = jax.tree.map(jnp.asarray, LegacyData.create_empty())
        self.dt = 0.0  # Time since the last correction

        self._correct = jax.jit(legacy_correct)
        # Compile now instead of when the first measurement arrives
        pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        self._correct(self.data, self.settings, pos, quat, np.float64(0.0))
        self._update_estimate()

    def predict(self, dt: float) -> EstimatorData:
        """This function is not part of the legacy estimator and only for compatability."""
        # Since the legacy estimator doesn't inherently support the prediction/correction
        # form of a Kalman filter, we only step the time in the prediction step. In the
        # correction step, we use the accumulated time step size to correct with the
        # correct dt. Therefore, the estimate itself only changes in the correction step.
        self.dt += dt
        return self._estimate

    def correct(self, pos: Array, quat: Array) -> EstimatorData:
        """This function is not part of the legacy estimator and only for compatability."""
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        self.data = self._correct(self.data, self.settings, pos, quat, np.float64(self.dt))
        self.dt = 0.0
        self._update_estimate()
        return self._estimate

    def set_state(self, pos: Array, quat: Array):
        """This function is not part of the legacy estimator and only for compatability."""
        pos = jnp.asarray(pos, dtype=jnp.float64)
        quat = jnp.asarray(quat, dtype=jnp.float64)
        # Also use the state as the previous measurement. Otherwise, the first numeric
        # derivatives are computed from zero and cause a huge spike in the velocities.
        self.data = self.data.replace(pos=pos, quat=quat, pos_prev=pos, quat_prev=quat)
        self._update_estimate()

    def _update_estimate(self):
        data = self.data
        pos, quat, vel, ang_vel = map(np.array, (data.pos, data.quat, data.vel, data.ang_vel))
        self._estimate = EstimatorData.create(pos, quat, vel, ang_vel)


def legacy_correct(
    data: LegacyData, settings: LegacySettings, pos: Array, quat: Array, dt: float
) -> LegacyData:
    """Correct the estimate with a new measurement.

    Args:
        data: The current state of the estimator.
        settings: The filter time constants.
        pos: The measured position.
        quat: The measured orientation.
        dt: The time since the last correction.

    Returns:
        The corrected state.
    """
    data = compute_numerical_derivatives(data, pos, quat, dt)
    data = prior_update(data, dt)
    return low_pass_filter(data, settings, pos, quat, dt)


def compute_numerical_derivatives(
    data: LegacyData, pos: Array, quat: Array, dt: float
) -> LegacyData:
    """Compute velocity, acceleration, and angular velocity from consecutive measurements."""
    xp = data.pos.__array_namespace__()

    # Skip impossibly small time differences. To stay jittable, we compute the derivatives with a
    # dummy dt and keep the previous values instead of returning early.
    valid = dt > 1e-15
    dt = xp.where(valid, dt, 1.0)
    vel_meas = (pos - data.pos_prev) / dt
    acc_meas = (vel_meas - data.vel_prev) / dt
    ang_vel_meas = (R.from_quat(data.quat_prev).inv() * R.from_quat(quat)).as_rotvec() / dt

    return data.replace(
        vel_meas=xp.where(valid, vel_meas, data.vel_meas),
        acc_meas=xp.where(valid, acc_meas, data.acc_meas),
        ang_vel_meas=xp.where(valid, ang_vel_meas, data.ang_vel_meas),
        pos_prev=xp.where(valid, pos, data.pos_prev),
        vel_prev=xp.where(valid, vel_meas, data.vel_prev),
        quat_prev=xp.where(valid, quat, data.quat_prev),
    )


def prior_update(data: LegacyData, dt: float) -> LegacyData:
    """Predict the state assuming constant acceleration and angular velocity."""
    pos = data.pos + dt * data.vel + 0.5 * dt * dt * data.acc
    vel = data.vel + dt * data.acc
    quat = (R.from_quat(data.quat) * R.from_rotvec(data.ang_vel * dt)).as_quat()
    return data.replace(pos=pos, vel=vel, quat=quat)


def low_pass_filter(
    data: LegacyData, settings: LegacySettings, pos: Array, quat: Array, dt: float
) -> LegacyData:
    """Move the predicted state towards the measurements and their numerical derivatives.

    Each quantity moves by the fraction 1 - exp(-dt / tau) of its difference to the measurement.
    """
    xp = data.pos.__array_namespace__()

    def weight(tau: float) -> Array:
        return 1.0 - xp.exp(-dt / tau)

    pos = data.pos + weight(settings.tau_est_trans) * (pos - data.pos)
    vel = data.vel + weight(settings.tau_est_trans_dot) * (data.vel_meas - data.vel)
    acc = data.acc + weight(settings.tau_est_trans_dot_dot) * (data.acc_meas - data.acc)
    ang_vel = data.ang_vel + weight(settings.tau_est_rot_dot) * (data.ang_vel_meas - data.ang_vel)
    # The orientation moves along the rotation from the predicted to the measured one
    rot = R.from_quat(data.quat)
    rotvec = weight(settings.tau_est_rot) * (rot.inv() * R.from_quat(quat)).as_rotvec()
    quat = (rot * R.from_rotvec(rotvec)).as_quat()
    return data.replace(pos=pos, vel=vel, acc=acc, quat=quat, ang_vel=ang_vel)
