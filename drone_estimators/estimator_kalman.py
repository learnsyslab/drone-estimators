"""Kalman filter estimators."""

from __future__ import absolute_import, annotations, division, print_function

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from drone_estimators.estimator import Estimator
from drone_estimators.filterpy import (
    Q_discrete_white_noise,
    ukf_correct,
    ukf_predict,
    ukf_predict_correct,
)
from drone_estimators.structs.estimator_data import EstimatorData
from drone_estimators.structs.ukf_data import SigmaPointsSettings, UKFSettings
from drone_estimators.utils.dynamics import dynamics_function, observation_function

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


class KalmanFilter(Estimator):
    """Unscented Kalman Filter class that wraps the filterPY toolbox."""

    def __init__(
        self,
        dt: float,
        model: str = "so_rpy",
        config: str = "cf2x_L250",
        filter_type: Literal["UKF"] = "UKF",
        estimate_rotor_vel: bool = False,
        estimate_dist_f: bool = False,
        estimate_dist_t: bool = False,
        initial_obs: dict[str, Array] | None = None,
        batch_shape: tuple[int, ...] = (),
        jit: bool = False,
    ):  # TODO give obs and info # analytical_mel_att
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
            model: The name of the model that is to be used.
            config: The setup configuration of the drone.
            filter_type: The type of Kalman filter. Only "UKF" is implemented so far.
            estimate_rotor_vel: If the rotor speeds should be estimated, defaults to False.
            estimate_dist_f: If the disturbance forces should be estimated, defaults to False.
            estimate_dist_t: If the disturbance torques should be estimated, defaults to False.
            initial_obs: Optional, the initial observation of the environment's state. See the environment's observation space for details.
            batch_shape: The batch dimensions, e.g., (n_drones,). All arrays have leading batch dimensions.
            jit: If the filter should be jit compiled with jax. Requires 64 bit precision.
        """
        assert filter_type == "UKF", f"Filter type {filter_type} is not implemented."
        fx = dynamics_function(model, config)
        hx = observation_function
        # fx = jax.jit(dynamics_numeric(model, config))
        # hx = jax.jit(observation_function)

        dim_x = 13
        if estimate_rotor_vel:
            dim_x += 4
        if estimate_dist_f:
            dim_x += 3
        if estimate_dist_t:
            dim_x += 3
        dim_u = 4  # TODO from available models
        dim_z = 7  # pos, quat

        self.data = EstimatorData.create_empty(
            rotor_vel=estimate_rotor_vel,
            dist_f=estimate_dist_f,
            dist_t=estimate_dist_t,
            dim_u=dim_u,
            dim_z=dim_z,
            batch_shape=batch_shape,
        )
        self.batch_shape = batch_shape

        dim_x = EstimatorData.get_state_dim(self.data)
        # print(f"dim_x={dim_x}")

        Q, R = self.create_covariance_matrices(
            dim_x=dim_x,
            dim_z=dim_z,
            varQ_pos=1e-6,
            varQ_quat=1e-2,
            varQ_rotor_vel=1e-5,
            varR_pos=1e-9,
            varR_quat=5e-9,
            dt=dt,
        )
        # Q = self.create_Q(
        #     dim_x=dim_x, varQ_pos=1e-6, varQ_quat=1e-6, varQ_vel=1e-3, varQ_ang_vel=1e-2, dt=dt
        # )
        # R = self.create_R(dim_z=dim_z, varR_pos=1e-8, varR_quat=3e-6, dt=dt)

        sigma_settings = SigmaPointsSettings.create(n=dim_x, alpha=1e-3, beta=2.0, kappa=0.0)
        self.settings = UKFSettings.create(SPsettings=sigma_settings, Q=Q, R=R, fx=fx, hx=hx)

        self._predict = partial(ukf_predict, settings=self.settings)
        self._correct = partial(ukf_correct, settings=self.settings)
        self._predict_correct = partial(ukf_predict_correct, settings=self.settings)
        if jit:
            # The covariances span many orders of magnitude, which 32 bit can't resolve
            if not jax.config.jax_enable_x64:
                raise RuntimeError(
                    "The jitted Kalman filter needs 64 bit precision. "
                    'Call jax.config.update("jax_enable_x64", True) before creating it.'
                )
            self.data = jax.tree.map(jnp.asarray, self.data)
            self._predict = jax.jit(self._predict)
            self._correct = jax.jit(self._correct)
            self._predict_correct = jax.jit(self._predict_correct)
            # Compile now instead of when the first measurement arrives
            mask = np.zeros(batch_shape, dtype=bool)
            self._correct(self._predict(self.data, mask=mask), mask=mask)

        # Initialize state and covariance
        if initial_obs is not None:
            self.set_state(initial_obs["pos"], initial_obs["quat"])

    def set_state(self, pos: Array, quat: Array):
        """Sets pos and quat of the state to the given values and reduces the covariance value.

        With aggressive Q and R values, this needs to be called in the beginning. Otherwise the estimator might fail.
        """
        self.data = self.data.replace(pos=pos, quat=quat)
        dim_x = EstimatorData.as_state_array(self.data).shape[-1]
        covariance = np.broadcast_to(np.eye(dim_x) * 1e-6, self.data.covariance.shape)
        self.data = self.data.replace(covariance=covariance)

    def create_covariance_matrices(
        self,
        dim_x: int,
        dim_z: int,
        varQ_pos: float,
        varQ_quat: float,
        varQ_rotor_vel: float,
        varR_pos: float,
        varR_quat: float,
        dt: float,
        varQ_dist_f: float = 1e-11,
        varQ_dist_t: float = 1e-18,  # TODO dist torque very noisy
    ) -> tuple[Array, Array]:
        """Creates sophisticated covariance matrices based on a paper.

        We assume that the linear elements (pos, vel) are correlated
        and the rotational elements (quat, ang_vel) are.
        For more information consult the filterpy library.
        """
        ### Set process noise covariance (tunable). Uncertainty in the dynamics. High Q -> less trust in model
        Q = np.eye(dim_x)

        # TODO maybe add correlation with rotor_vel!

        # Linear equations
        Q_xyz = Q_discrete_white_noise(
            dim=2, dt=dt, var=varQ_pos, block_size=3, order_by_dim=False
        )  # pos & vel
        Q[0:3, 0:3] = Q_xyz[0:3, 0:3]  # pos
        Q[7:10, 7:10] = Q_xyz[3:6, 3:6]  # vel
        Q[0:3, 7:10] = Q_xyz[0:3, 3:6]  # pos <-> vel
        Q[7:10, 0:3] = Q_xyz[3:6, 0:3]  # pos <-> vel

        # Rotational equations
        # since quat and ang_vel have different length, it's a little more tricky
        Q_rot = Q_discrete_white_noise(
            dim=2, dt=dt, var=varQ_quat, block_size=3, order_by_dim=False
        )  # quat & ang_vel
        Q[3:5, 3:5] = Q_rot[0:2, 0:2]  # quat12
        Q[5:7, 5:7] = Q_rot[0:2, 0:2]  # quat34
        Q[10:13, 10:13] = Q_rot[3:6, 3:6]  # ang_vel
        # We know that all quat_dot dependent on all ang_vel
        # => fill in the whole 4x3 and 3x4 matrix blocks with the variance
        Q[3:7, 10:13] = Q_rot[0, 3]  # quat <-> ang_vel
        Q[10:13, 3:7] = Q_rot[3, 0]  # quat <-> ang_vel

        i = 13  # for keeping how big the state (index) is

        # Motor forces TODO: maybe add correlation of acceleration (and angular acc) to motor forces
        if self.data.rotor_vel is not None:
            Q_forces = Q_discrete_white_noise(
                dim=2, dt=dt, var=varQ_rotor_vel, block_size=1, order_by_dim=False
            )  # Motor Forces
            Q[i : i + 2, i : i + 2] = np.eye(2) * Q_forces[0, 0]  # forces 1&2
            Q[i + 2 : i + 4, i + 2 : i + 4] = np.eye(2) * Q_forces[0, 0]  # forces 3&4
            Q[7:10, i : i + 4] = Q_forces[0, 1]  # forces <-> vel
            Q[i : i + 4, 7:10] = Q_forces[1, 0]  # forces <-> vel
            Q[10:13, i : i + 4] = (
                Q_forces[0, 1] * 0.04
            )  # forces <-> ang_vel, times arm length (F=rxl)
            Q[i : i + 4, 10:13] = (
                Q_forces[1, 0] * 0.04
            )  # forces <-> ang_vel, times arm length (F=rxl)
            # Q[i : i + 4] *= varQ_rotor_vel  # TODO move index as in dataclass example and make optional
            i = i + 4

        # External forces and torques TODO Maybe make x and v dependend on F uncertainty and same for torque
        if self.data.dist_f is not None:  # TODO move index as in dataclass example
            Q[i : i + 3, i : i + 3] *= varQ_dist_f  # Force
            i = i + 3
        if self.data.dist_t is not None:
            Q[i : i + 3, i : i + 3] *= varQ_dist_t  # Torque

        ### Set measurement noise covariance (tunable). Uncertaints in the measurements. High R -> less trust in measurements
        R = np.eye(dim_z)  # Assuming uncorrelated noise
        # very low noise on the position ("mm precision" => even less noise)
        R[:3, :3] = R[:3, :3] * varR_pos
        # "high" measurements noise on the angles, estimate: 0.01 constains all values => std=3e-3 TODO look at new quat measurements
        R[3:, 3:] = R[3:, 3:] * varR_quat

        return Q, R

    def create_Q(
        self,
        dim_x: int,
        varQ_pos: float,
        varQ_quat: float,
        varQ_vel: float,
        varQ_ang_vel: float,
        dt: float,
        varQ_rotor_vel: float = 1e-1,
        varQ_dist_f: float = 1e-9,
        varQ_dist_t: float = 1e-12,
    ) -> Array:
        """TODO."""
        Q = np.eye(dim_x)
        # TODO remove, just for testing
        Q[0:3] *= varQ_pos
        Q[3:7] *= varQ_quat
        Q[7:10] *= varQ_vel
        Q[10:13] *= varQ_ang_vel
        i = 13
        if self.data.rotor_vel is not None:
            Q[i : i + 4] *= varQ_rotor_vel
            i = i + 4
        if self.data.dist_f is not None:
            Q[i : i + 3, i : i + 3] *= varQ_dist_f  # Force
            i = i + 3
        if self.data.dist_t is not None:
            Q[i : i + 3, i : i + 3] *= varQ_dist_t  # Torque
        return Q

    def create_R(self, dim_z: int, varR_pos: float, varR_quat: float, dt: float) -> Array:
        """TODO."""
        ### Set measurement noise covariance (tunable). Uncertaints in the measurements. High R -> less trust in measurements
        R = np.eye(dim_z)  # Assuming uncorrelated noise
        # very low noise on the position ("mm precision" => even less noise)
        R[:3, :3] = R[:3, :3] * varR_pos
        # "high" measurements noise on the angles, estimate: 0.01 constains all values => std=3e-3 TODO look at new quat measurements
        R[3:, 3:] = R[3:, 3:] * varR_quat
        return R

    def step(
        self, pos: Array, quat: Array, dt: float, command: Array | None = None
    ) -> EstimatorData:
        """Steps the UKF by one. Doing one prediction and correction step.

        Args:
            pos: Latest observation of the position
            quat: Latest observation of the quaternion
            dt: Optional, time step size. If not specified, default time is used
            command: Optional, latest input to the system

        Return:
            New state prediction
        """
        # Check if time step is positive. Only batch elements with a positive time step are updated
        dt = np.broadcast_to(np.asarray(dt, dtype=np.float64), self.batch_shape)
        if not np.any(dt > 0):
            return self.data

        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(z=np.concat((pos, quat), axis=-1), dt=dt)

        # if self.data.dt > 0:  # TODO make dt check more elegant and catch all errors
        # self.data = ukf_predict(self.data, self.settings)
        # self.data = ukf_predict(self.data, self.settings)
        self.data = self._predict_correct(self.data, mask=dt > 0)

        return self.data

    def predict(self, dt: float, command: Array | None = None) -> EstimatorData:
        """TODO."""
        # Check if time step is positive. Only batch elements with a positive time step are updated
        dt = np.broadcast_to(np.asarray(dt, dtype=np.float64), self.batch_shape)
        if not np.any(dt > 0):
            return self.data

        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(dt=dt)

        self.data = self._predict(self.data, mask=dt > 0)

        return self.data

    def correct(
        self, pos: Array, quat: Array, command: Array | None = None, mask: Array | None = None
    ) -> EstimatorData:
        """TODO. If given, only batch elements where mask is True are corrected."""
        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(z=np.concat((pos, quat), axis=-1))

        # print(f"{quat=}, {self.data.quat=}, {self.data.u[-2]=}")

        self.data = self._correct(self.data, mask=mask)

        return self.data
