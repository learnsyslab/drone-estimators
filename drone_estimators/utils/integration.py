"""Based on https://github.com/utiasDSL/crazyflow/blob/d38bb5c75fe6624972ccc18d89789d3636cfd8cd/crazyflow/sim/integration.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later
    from drone_estimators.structs.estimator_data import EstimatorData


def integrate_EstimatorData(state: EstimatorData, state_dot: EstimatorData) -> EstimatorData:
    """Integrates EstimatorData properly."""
    next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel = _integrate(
        state.pos,
        state.quat,
        state.vel,
        state.ang_vel,
        state_dot.pos,
        state_dot.quat,
        state_dot.vel,
        state_dot.ang_vel,
        state.dt,
        state.rotor_vel,
        state_dot.rotor_vel,
    )
    # TODO unit norm of quaternion!
    # TODO implement different integrator types later
    return state.replace(
        pos=next_pos, quat=next_quat, vel=next_vel, ang_vel=next_ang_vel, rotor_vel=next_rotor_vel
    )


def _integrate(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    pos_dot: Array,
    quat_dot: Array,
    vel_dot: Array,
    ang_vel_dot: Array,
    dt: float,
    rotor_vel: Array | None = None,
    rotor_vel_dot: Array | None = None,
) -> Array:  # TODO is actually tuple
    """Integrate the dynamics forward in time.

    Args:
        pos: The position of the drone.
        quat: The orientation of the drone as a quaternion.
        vel: The velocity of the drone.
        ang_vel: The angular velocity of the drone.
        pos_dot: The derivative of the position of the drone.
        quat_dot: The derivative of the quaternion of the drone.
        vel_dot: The derivative of the velocity of the drone.
        ang_vel_dot: The derivative of the angular velocity of the drone.
        dt: The time step to integrate over.
        rotor_vel: The forces for the motors.
        rotor_vel_dot: The derivative of the motor forces.

    Returns:
        The next position, quaternion, velocity, and roll, pitch, and yaw rates of the drone.
    """
    next_pos = pos + pos_dot * dt
    next_quat = (R.from_quat(quat) * R.from_rotvec(ang_vel * dt)).as_quat()
    next_vel = vel + vel_dot * dt
    next_ang_vel = ang_vel + ang_vel_dot * dt
    next_rotor_vel = None
    if rotor_vel is not None:
        next_rotor_vel = rotor_vel + rotor_vel_dot * dt

    return next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel
