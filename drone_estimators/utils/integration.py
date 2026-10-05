"""Based on https://github.com/learnsyslab/crazyflow/blob/d28ec70d2478a230f1feb0fb3d644d1bbdbb8107/crazyflow/sim/integration.py.

Unlike crazyflow's version, this works with any array API backend and without rotor velocities.
"""

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
        state.rotor_vel,
        state_dot.pos,
        state.ang_vel,  # The orientation changes with the angular velocity
        state_dot.vel,
        state_dot.ang_vel,
        state_dot.rotor_vel,
        state.dt,
    )
    # TODO implement different integrator types later
    return state.replace(
        pos=next_pos, quat=next_quat, vel=next_vel, ang_vel=next_ang_vel, rotor_vel=next_rotor_vel
    )


def _integrate(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    rotor_vel: Array | None,
    dpos: Array,
    drot: Array,
    dvel: Array,
    dang_vel: Array,
    drotor_vel: Array | None,
    dt: float,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """Integrate the dynamics forward in time.

    Args:
        pos: The position of the drone.
        quat: The orientation of the drone as a quaternion.
        vel: The velocity of the drone.
        ang_vel: The angular velocity of the drone.
        rotor_vel: The rotor velocity of the drone, None if not estimated.
        dpos: The derivative of the position of the drone.
        drot: The derivative of the quaternion of the drone (3D angular velocity).
        dvel: The derivative of the velocity of the drone.
        dang_vel: The derivative of the angular velocity of the drone.
        drotor_vel: The derivative of the rotor velocity of the drone, None if not estimated.
        dt: The time step to integrate over.

    Returns:
        The next position, quaternion, velocity, angular velocity, and rotor velocity of the drone.
    """
    xp = pos.__array_namespace__()
    next_pos = pos + dpos * dt
    # Prevent NaN gradients by setting extremely small rotations to 0, as crazyflow does
    drot = xp.where(xp.abs(drot) < xp.finfo(drot.dtype).smallest_normal, 0.0, drot)
    next_quat = (R.from_quat(quat) * R.from_rotvec(drot * dt)).as_quat()
    next_vel = vel + dvel * dt
    next_ang_vel = ang_vel + dang_vel * dt
    next_rotor_vel = None if rotor_vel is None else rotor_vel + drotor_vel * dt
    return next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel
