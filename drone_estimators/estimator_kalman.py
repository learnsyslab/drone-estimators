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
    ekf_correct,
    ekf_predict,
    ekf_predict_correct,
    ukf_correct,
    ukf_predict,
    ukf_predict_correct,
)
from drone_estimators.structs.ekf_data import EKFSettings
from drone_estimators.structs.estimator_data import EstimatorData
from drone_estimators.structs.ukf_data import SigmaPointsSettings, UKFSettings
from drone_estimators.utils.dynamics import dynamics_function, hover_jacobians, observation_function

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


class KalmanFilter(Estimator):
    """Extended or unscented Kalman Filter class that wraps the filterPY toolbox."""

    def __init__(
        self,
        dt: float,
        model: str = "so_rpy",
        config: str = "cf2x_L250",
        filter_type: Literal["EKF", "UKF"] = "UKF",
        estimate_rotor_vel: bool = False,
        estimate_dist_f: bool = False,
        estimate_dist_t: bool = False,
        initial_obs: dict[str, Array] | None = None,
        batch_shape: tuple[int, ...] = (),
        jit: bool = False,
    ):
        """Initialize basic parameters.

        Args:
            dt: Time step between callings.
            model: The name of the model that is to be used.
            config: The setup configuration of the drone.
            filter_type: The type of Kalman filter, "EKF" or "UKF". The EKF uses the Jacobian
                of the dynamics at hover.
            estimate_rotor_vel: If the rotor speeds should be estimated, defaults to False.
            estimate_dist_f: If the disturbance forces should be estimated, defaults to False.
            estimate_dist_t: If the disturbance torques should be estimated, defaults to False.
            initial_obs: Optional, the initial observation of the environment's state. See the environment's observation space for details.
            batch_shape: The batch dimensions, e.g., (n_drones,). All arrays have leading batch dimensions.
            jit: If the filter should be jit compiled with jax.
        """
        fx = dynamics_function(model, config)
        hx = observation_function

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

        Q, R = self.create_covariance_matrices(
            dim_x=dim_x,
            dim_z=dim_z,
            varQ_pos=1e-3,
            varQ_quat=10.0,
            varQ_rotor_vel=1e-5,
            varR_pos=1e-7,
            varR_quat=1e-5,
            dt=dt,
            varQ_dist_f=1e-8,
            varQ_dist_t=1e-13,
        )

        match filter_type:
            case "EKF":
                A, H = hover_jacobians(
                    model, config, estimate_rotor_vel, estimate_dist_f, estimate_dist_t
                )
                self.settings = EKFSettings.create(A=A, H=H, Q=Q, R=R, fx=fx, hx=hx)
                predict, correct, predict_correct = ekf_predict, ekf_correct, ekf_predict_correct
            case "UKF":
                # alpha = 1 and kappa = 0 keeps the unscented transform well conditioned in 32 bit
                sigma_settings = SigmaPointsSettings.create(n=dim_x, alpha=1.0, beta=2.0, kappa=0.0)
                self.settings = UKFSettings.create(
                    SPsettings=sigma_settings, Q=Q, R=R, fx=fx, hx=hx
                )
                predict, correct, predict_correct = ukf_predict, ukf_correct, ukf_predict_correct
            case _:
                raise ValueError(f"Filter type {filter_type} is not implemented.")

        self._predict = partial(predict, settings=self.settings)
        self._correct = partial(correct, settings=self.settings)
        self._predict_correct = partial(predict_correct, settings=self.settings)
        if jit:
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

        Args:
            pos: The position the state is set to.
            quat: The quaternion the state is set to.
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
        varQ_dist_f: float = 1e-8,
        varQ_dist_t: float = 1e-13,
    ) -> tuple[Array, Array]:
        """Creates sophisticated covariance matrices based on a paper.

        We assume that the linear elements (pos, vel) are correlated
        and the rotational elements (quat, ang_vel) are.
        For more information consult the filterpy library.

        Args:
            dim_x: The dimension of the state.
            dim_z: The dimension of the measurement.
            varQ_pos: The variance of the position and velocity process noise.
            varQ_quat: The variance of the quaternion and angular velocity process noise.
            varQ_rotor_vel: The variance of the rotor velocity process noise, if estimated.
            varR_pos: The variance of the position measurement noise.
            varR_quat: The variance of the quaternion measurement noise.
            dt: The time step.
            varQ_dist_f: The variance of the disturbance force process noise, if estimated.
            varQ_dist_t: The variance of the disturbance torque process noise, if estimated.

        Returns:
            The process noise covariance Q of shape (dim_x, dim_x) and the measurement noise
            covariance R of shape (dim_z, dim_z).
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
        # quat and ang_vel have different lengths and the coupling quat_dot = 0.5 * quat x ang_vel
        # depends on the state, so they are not correlated here: filling the whole 4x3 block with
        # the cross term of the white noise model makes Q indefinite, which breaks the Cholesky
        # decomposition of the sigma points once varQ_quat is large enough to matter.
        Q_rot = Q_discrete_white_noise(
            dim=2, dt=dt, var=varQ_quat, block_size=3, order_by_dim=False
        )  # quat & ang_vel
        Q[3:7, 3:7] = np.eye(4) * Q_rot[0, 0]  # quat
        Q[10:13, 10:13] = Q_rot[3:6, 3:6]  # ang_vel

        i = 13  # for keeping how big the state (index) is

        # Rotor velocities. They are not correlated with the velocities: the thrust acts along the
        # body z axis only, and a dense cross block with the velocities makes Q indefinite.
        if self.data.rotor_vel is not None:
            Q[i : i + 4, i : i + 4] = np.eye(4) * varQ_rotor_vel * dt**2
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

        eig_min = np.linalg.eigvalsh(Q).min()  # Only round-off below zero
        assert eig_min >= -1e-9 * np.abs(Q).max(), f"The process noise is indefinite ({eig_min=})"
        return Q, R

    def step(
        self, pos: Array, quat: Array, dt: float, command: Array | None = None
    ) -> EstimatorData:
        """Steps the filter by one. Doing one prediction and correction step.

        Args:
            pos: Latest observation of the position.
            quat: Latest observation of the quaternion.
            dt: Time step size. Only batch elements with a positive dt are updated.
            command: Optional, latest input to the system.

        Returns:
            New state estimate.
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

        self.data = self._predict_correct(self.data, mask=dt > 0)

        return self.data

    def predict(self, dt: float, command: Array | None = None) -> EstimatorData:
        """Predict the state by one time step. Only batch elements with a positive dt are updated.

        Args:
            dt: Time step size.
            command: Optional, latest input to the system.

        Returns:
            New state prediction.
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
        self.data = self.data.replace(dt=dt)

        self.data = self._predict(self.data, mask=dt > 0)

        return self.data

    def correct(
        self, pos: Array, quat: Array, command: Array | None = None, mask: Array | None = None
    ) -> EstimatorData:
        """Correct the state with the latest observation.

        Args:
            pos: Latest observation of the position.
            quat: Latest observation of the quaternion.
            command: Optional, latest input to the system.
            mask: Optional, only batch elements where mask is True are corrected.

        Returns:
            New state estimate.
        """
        # Update the input
        if command is not None:
            self.set_input(command)

        # Update observation and dt
        # dt hast to be vectorized to work properly in jax
        self.data = self.data.replace(z=np.concat((pos, quat), axis=-1))

        self.data = self._correct(self.data, mask=mask)

        return self.data
