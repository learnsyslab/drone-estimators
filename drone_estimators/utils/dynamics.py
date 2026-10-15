"""This file implements some wrappers to get the dynamics from crazyflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, available_dynamics, dynamics_features, parametrize

if TYPE_CHECKING:
    from collections.abc import Callable

    from drone_estimators._typing import Array  # To be changed to array_api_typing later


def dynamics_function(model: str, config: str) -> Callable:
    """Return the crazyflow dynamics of the model with the parameters of the drone config.

    Models without rotor dynamics (e.g. so_rpy) neither take rotor_vel nor return its derivative.
    The returned function always does, such that the estimators can call all models the same way.
    """
    fn = parametrize(available_dynamics[Dynamics(model)], Drone(config))
    if dynamics_features(fn)["rotor_dynamics"]:
        return fn

    def fn_without_rotor_dynamics(
        *args: Array, rotor_vel: Array | None = None, **kwargs: Array | None
    ) -> tuple[Array | None, ...]:
        assert rotor_vel is None, f"Model {model} has no rotor dynamics"
        return *fn(*args, **kwargs), None

    fn_without_rotor_dynamics.__dynamics_features__ = {"rotor_dynamics": False}
    return fn_without_rotor_dynamics


def observation_function(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
) -> Array:
    """Return the observable part of the state.

    This is basically not necessary, since we always get position and orientation
    from Vicon. However, for sake of completeness, this observation function is added.
    """
    xp = pos.__array_namespace__()
    return xp.concat((pos, quat), axis=-1)
