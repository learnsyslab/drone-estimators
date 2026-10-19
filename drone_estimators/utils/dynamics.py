"""This file implements some wrappers to get the dynamics from crazyflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import Drone
from crazyflow.dynamics import (
    Dynamics,
    available_dynamics,
    dynamics_features,
    load_params,
    parametrize,
)

from drone_estimators.structs.estimator_data import EstimatorData

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


def hover_jacobians(
    model: str,
    config: str,
    estimate_rotor_vel: bool = False,
    estimate_dist_f: bool = False,
    estimate_dist_t: bool = False,
) -> tuple[Array, Array]:
    """Return the Jacobians of the dynamics and the observation function at hover.

    At hover, the drone is level and at rest, the command compensates gravity, and the rotors are at
    their steady state. The disturbances are modeled as constant.

    Args:
        model: The name of the model that is to be used.
        config: The setup configuration of the drone.
        estimate_rotor_vel: If the rotor speeds are part of the state.
        estimate_dist_f: If the disturbance forces are part of the state.
        estimate_dist_t: If the disturbance torques are part of the state.

    Returns:
        The Jacobians of the continuous dynamics and of the observation function w.r.t. the state.
    """
    params = load_params(Dynamics(model), Drone(config))
    weight = -params["mass"] * params["gravity_vec"][-1]
    if Dynamics(model) == Dynamics.first_principles:  # Commands the motor speeds (RPM)
        cmd = motor_force2rotor_vel(np.full(4, weight / 4), params["rpm2thrust"])
        rotor_vel = cmd
    else:  # Commands roll, pitch, yaw (rad) and thrust (N), which is also the rotor state
        thrust = (weight - params["acc_coef"]) / params["cmd_f_coef"]
        cmd, rotor_vel = np.array([0.0, 0.0, 0.0, thrust]), np.full(4, thrust)
    data = EstimatorData.create_empty(
        rotor_vel=estimate_rotor_vel, dist_f=estimate_dist_f, dist_t=estimate_dist_t
    )
    if estimate_rotor_vel:
        data = data.replace(rotor_vel=rotor_vel)
    fx = dynamics_function(model, config)

    def state_dot(x: Array) -> Array:
        s = EstimatorData.from_state_array(data, x)
        derivatives = fx(
            pos=s.pos,
            quat=s.quat,
            vel=s.vel,
            ang_vel=s.ang_vel,
            cmd=cmd,
            rotor_vel=s.rotor_vel,
            dist_f=s.dist_f,
            dist_t=s.dist_t,
        )
        x_dot = jnp.concat([d for d in derivatives if d is not None])
        # The disturbances are constant
        return jnp.concat((x_dot, jnp.zeros(x.shape[-1] - x_dot.shape[-1])))

    def observation(x: Array) -> Array:
        s = EstimatorData.from_state_array(data, x)
        return observation_function(
            s.pos, s.quat, s.vel, s.ang_vel, cmd, s.rotor_vel, s.dist_f, s.dist_t
        )

    x = EstimatorData.as_state_array(data)
    with jax.enable_x64(True):  # Also compute the Jacobians in 64 bit if jax runs in 32 bit
        return np.asarray(jax.jacfwd(state_dot)(x)), np.asarray(jax.jacfwd(observation)(x))
