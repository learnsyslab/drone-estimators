"""This file contains modified methods and classes taken from the filterpy library.

Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import absolute_import, annotations, division, print_function

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag

from drone_estimators.structs.estimator_data import EstimatorData
from drone_estimators.utils.batching import select
from drone_estimators.utils.integration import integrate_EstimatorData

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later
    from drone_estimators.structs.ekf_data import EKFSettings
    from drone_estimators.structs.ukf_data import UKFSettings


def ekf_predict_correct(
    data: EstimatorData, settings: EKFSettings, mask: Array | None = None
) -> EstimatorData:
    """Predict and correct the state.

    Args:
        data: The current state of the estimator, including the input and the measurement.
        settings: The filter settings.
        mask: Optional, only batch elements where mask is True are updated.

    Returns:
        The corrected state.
    """
    return ekf_correct(ekf_predict(data, settings, mask), settings, mask)


def ekf_predict(
    data: EstimatorData, settings: EKFSettings, mask: Array | None = None
) -> EstimatorData:
    """Predict the state and covariance.

    Args:
        data: The current state of the estimator, including the input and the time step.
        settings: The filter settings.
        mask: Optional, only batch elements where mask is True are updated.

    Returns:
        The predicted state.
    """
    xp = data.pos.__array_namespace__()
    data_in = data
    #### Predict
    # Pass the state through the dynamics
    pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = settings.fx(
        pos=data.pos,
        quat=data.quat,
        vel=data.vel,
        ang_vel=data.ang_vel,
        cmd=data.u,
        rotor_vel=data.rotor_vel,
        dist_f=data.dist_f,
        dist_t=data.dist_t,
    )
    data_dot = data.replace(
        pos=pos_dot, quat=quat_dot, vel=vel_dot, ang_vel=ang_vel_dot, rotor_vel=rotor_vel_dot
    )

    # Integrate dynamics if continuous. dt needs a trailing axis to broadcast against the states
    data_f = integrate_EstimatorData(data.replace(dt=data.dt[..., None]), data_dot)
    data = data_f.replace(dt=data.dt)

    # Propagate the covariance with the Jacobian of the dynamics, discretized with Euler
    F = xp.eye(settings.A.shape[-1]) + settings.A * data.dt[..., None, None]
    P = F @ data.covariance @ F.mT + settings.Q

    # save prior
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ekf_correct(
    data: EstimatorData, settings: EKFSettings, mask: Array | None = None
) -> EstimatorData:
    """Correct the state with the measurement.

    Args:
        data: The predicted state of the estimator, including the measurement.
        settings: The filter settings.
        mask: Optional, only batch elements where mask is True are updated.

    Returns:
        The corrected state.
    """
    xp = data.covariance.__array_namespace__()
    data_in = data
    H = settings.H
    PHT = data.covariance @ H.mT

    # K @ S = PHT => K = PHT @ S^-1 => or: S.T @ K.T = PHT.T
    S = H @ PHT + settings.R
    K = xp.linalg.solve(S.mT, PHT.mT).mT  # Kalman gain

    # Pass the state through the measurement function h(x,u) to get the residual
    zp = settings.hx(
        pos=data.pos,
        quat=data.quat,
        vel=data.vel,
        ang_vel=data.ang_vel,
        cmd=data.u,
        rotor_vel=data.rotor_vel,
        dist_f=data.dist_f,
        dist_t=data.dist_t,
    )
    y = xp.subtract(data.z, zp)  # residual

    # Update Gaussian state estimate (x, P)
    x = EstimatorData.as_state_array(data) + (K @ y[..., None])[..., 0]
    # P = (I-KH)P(I-KH)' + KRK' is more numerically stable than P = (I-KH)P
    I_KH = xp.eye(H.shape[-1]) - K @ H
    P = I_KH @ data.covariance @ I_KH.mT + K @ settings.R @ K.mT

    # Save posterior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ukf_predict_correct(
    data: EstimatorData, settings: UKFSettings, mask: Array | None = None
) -> EstimatorData:
    """Predict and correct the state.

    Args:
        data: The current state of the estimator, including the input and the measurement.
        settings: The filter settings.
        mask: Optional, only batch elements where mask is True are updated.

    Returns:
        The corrected state.
    """
    xp = data.pos.__array_namespace__()
    data_in = data
    #### Predict
    # Calculate sigma points
    # TODO special sigma points for quaternions!
    sigmas = ukf_calculate_sigma_points(data, settings)
    data_sigmas = EstimatorData.from_state_array(data, sigmas)
    data_sigmas = data_sigmas.replace(dt=data.dt[..., None, None])

    # Pass sigma points through dynamics
    pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = settings.fx(
        pos=data_sigmas.pos,
        quat=data_sigmas.quat,
        vel=data_sigmas.vel,
        ang_vel=data_sigmas.ang_vel,
        cmd=data.u[..., None, :],
        rotor_vel=data_sigmas.rotor_vel,
        dist_f=data_sigmas.dist_f,
        dist_t=data_sigmas.dist_t,
    )
    data_sigmas_dot = data_sigmas.replace(
        pos=pos_dot, quat=quat_dot, vel=vel_dot, ang_vel=ang_vel_dot, rotor_vel=rotor_vel_dot
    )

    # Integrate dynamics if continuous
    data_sigmas_f = integrate_EstimatorData(data_sigmas, data_sigmas_dot)
    sigmas_f = EstimatorData.as_state_array(data_sigmas_f)
    data = data.replace(sigmas_f=sigmas_f)

    # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
    sigmas_h = settings.hx(
        pos=data_sigmas_f.pos,
        quat=data_sigmas_f.quat,
        vel=data_sigmas_f.vel,
        ang_vel=data_sigmas_f.ang_vel,
        cmd=data.u[..., None, :],
        rotor_vel=data_sigmas_f.rotor_vel,
        dist_f=data_sigmas_f.dist_f,
        dist_t=data_sigmas_f.dist_t,
    )
    data = data.replace(sigmas_h=sigmas_h)

    # Compute prior with unscented transform
    x, P = ukf_unscented_transform(
        sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
    )

    # save prior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    #### Correct
    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )

    # compute cross variance
    Pxz = ukf_cross_variance(
        EstimatorData.as_state_array(data), zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc
    )
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.mT, Pxz.mT).mT  # Kalman gain
    y = xp.subtract(data.z, zp)  # residual

    # Update Gaussian state estimate (x, P)
    x = EstimatorData.as_state_array(data) + (K @ y[..., None])[..., 0]
    P = data.covariance - K @ S @ K.mT

    # Save posterior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ukf_predict(
    data: EstimatorData, settings: UKFSettings, mask: Array | None = None
) -> EstimatorData:
    """Predict the state and covariance.

    Args:
        data: The current state of the estimator, including the input and the time step.
        settings: The filter settings.
        mask: Optional, only batch elements where mask is True are updated.

    Returns:
        The predicted state.
    """
    data_in = data
    #### Predict
    # Calculate sigma points
    # TODO special sigma points for quaternions!
    sigmas = ukf_calculate_sigma_points(data, settings)
    data_sigmas = EstimatorData.from_state_array(data, sigmas)
    data_sigmas = data_sigmas.replace(dt=data.dt[..., None, None])

    # Pass sigma points through dynamics
    pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = settings.fx(
        pos=data_sigmas.pos,
        quat=data_sigmas.quat,
        vel=data_sigmas.vel,
        ang_vel=data_sigmas.ang_vel,
        cmd=data.u[..., None, :],
        rotor_vel=data_sigmas.rotor_vel,
        dist_f=data_sigmas.dist_f,
        dist_t=data_sigmas.dist_t,
    )
    data_sigmas_dot = data_sigmas.replace(
        pos=pos_dot, quat=quat_dot, vel=vel_dot, ang_vel=ang_vel_dot, rotor_vel=rotor_vel_dot
    )

    # Integrate dynamics if continuous
    data_sigmas_f = integrate_EstimatorData(data_sigmas, data_sigmas_dot)
    sigmas_f = EstimatorData.as_state_array(data_sigmas_f)
    data = data.replace(sigmas_f=sigmas_f)

    # Pass prior sigmas through measurment function h(x,u,dt) to get measurement sigmas
    sigmas_h = settings.hx(
        pos=data_sigmas_f.pos,
        quat=data_sigmas_f.quat,
        vel=data_sigmas_f.vel,
        ang_vel=data_sigmas_f.ang_vel,
        cmd=data.u[..., None, :],
        rotor_vel=data_sigmas_f.rotor_vel,
        dist_f=data_sigmas_f.dist_f,
        dist_t=data_sigmas_f.dist_t,
    )
    data = data.replace(sigmas_h=sigmas_h)

    # Compute prior with unscented transform
    x, P = ukf_unscented_transform(
        sigmas_f, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.Q
    )

    # save prior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ukf_correct(
    data: EstimatorData, settings: UKFSettings, mask: Array | None = None
) -> EstimatorData:
    """Correct the state with the measurement.

    Args:
        data: The predicted state of the estimator, including the measurement.
        settings: The filter settings.
        mask: Optional, only batch elements where mask is True are updated.

    Returns:
        The corrected state.
    """
    xp = data.covariance.__array_namespace__()
    data_in = data
    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )

    # compute cross variance
    Pxz = ukf_cross_variance(
        EstimatorData.as_state_array(data), zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc
    )
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.mT, Pxz.mT).mT  # Kalman gain
    y = xp.subtract(data.z, zp)  # residual

    # Update Gaussian state estimate (x, P)
    x = EstimatorData.as_state_array(data) + (K @ y[..., None])[..., 0]
    P = data.covariance - K @ S @ K.mT

    # Save posterior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ukf_calculate_sigma_points(data: EstimatorData, settings: UKFSettings) -> Array:
    """Compute the sigma points of the state with the Van der Merwe scaling.

    Args:
        data: The current state of the estimator.
        settings: The filter settings, including the sigma point parameters.

    Returns:
        The 2n+1 sigma points, stacked along the second to last axis.
    """
    xp = data.pos.__array_namespace__()
    P = data.covariance
    # Adding some very small identity part for numerical stability
    # Note: Higher values make the system more stable for the cost of more noise!
    P = P + xp.eye(P.shape[-1]) * 1e-12
    U = xp.linalg.cholesky((settings.SPsettings.lambda_ + settings.SPsettings.n) * P, upper=True)

    state_array = EstimatorData.as_state_array(data)[..., None, :]
    sigma_center = state_array
    # TODO for the quaternions use more sophisitcated approach to keep length 1!
    # Mainly: rotate in tangent space
    sigma_pos = xp.subtract(state_array, -U)
    sigma_neg = xp.subtract(state_array, U)
    return xp.concat((sigma_center, sigma_pos, sigma_neg), axis=-2)


def ukf_unscented_transform(
    sigmas: Array, Wm: Array, Wc: Array, noise_cov: Array = None
) -> tuple[Array, Array]:
    """Compute the mean and covariance of the sigma points.

    Args:
        sigmas: The sigma points, stacked along the second to last axis.
        Wm: The weights for the mean.
        Wc: The weights for the covariance.
        noise_cov: Optional, noise covariance that is added to the result.

    Returns:
        The mean and covariance.
    """
    xp = sigmas.__array_namespace__()
    x = Wm @ sigmas

    # new covariance is the sum of the outer product
    # of the residuals times the weights
    y = sigmas - x[..., None, :]
    P = y.mT @ (xp.diag(Wc) @ y)

    if noise_cov is not None:
        P = P + noise_cov

    return (x, P)


def ukf_cross_variance(x: Array, z: Array, sigmas_f: Array, sigmas_h: Array, Wc: Array) -> Array:
    """Compute cross variance of the state `x` and measurement `z`.

    Args:
        x: The mean of the state sigma points.
        z: The mean of the measurement sigma points.
        sigmas_f: The state sigma points.
        sigmas_h: The measurement sigma points.
        Wc: The weights for the covariance.

    Returns:
        The cross variance of shape (dim_x, dim_z).
    """
    xp = x.__array_namespace__()
    # The slicing brings Wc in the correct shape to be broadcast
    # The einsum as set up here takes the outer product of all the stacked vectors
    Pxz = Wc[:, None, None] * xp.einsum(
        "...bi,...bj->...bij",
        xp.subtract(sigmas_f, x[..., None, :]),
        xp.subtract(sigmas_h, z[..., None, :]),
    )
    Pxz = xp.sum(Pxz, axis=-3)
    return Pxz


def order_by_derivative(Q: Array, dim: int, block_size: int) -> Array:
    """Reorder a block diagonal noise matrix from being ordered by dimension to by derivative.

    Args:
        Q: The noise matrix of one dimension, of shape (dim, dim).
        dim: The number of derivatives in Q.
        block_size: The number of dimensions, e.g., 3 for x, y, z.

    Returns:
        The reordered noise matrix of shape (dim * block_size, dim * block_size).
    """
    xp = Q.__array_namespace__()

    N = dim * block_size

    D = xp.zeros((N, N))

    Q = xp.array(Q)
    for i, x in enumerate(Q.ravel()):
        f = xp.eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix : ix + block_size, iy : iy + block_size] = f

    return D


def Q_discrete_white_noise(
    dim: int, dt: float = 1.0, var: float = 1.0, block_size: int = 1, order_by_dim: bool = True
) -> Array:
    """Create the discrete white noise covariance for a constant derivative model.

    Args:
        dim: The number of derivatives, 2, 3, or 4, e.g., 2 for position and velocity.
        dt: The time step.
        var: The variance of the white noise.
        block_size: The number of dimensions, e.g., 3 for x, y, z.
        order_by_dim: If the matrix is ordered by dimension, else by derivative.

    Returns:
        The noise covariance of shape (dim * block_size, dim * block_size).
    """
    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]
    elif dim == 3:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1],
        ]
    else:
        Q = [
            [(dt**6) / 36, (dt**5) / 12, (dt**4) / 6, (dt**3) / 6],
            [(dt**5) / 12, (dt**4) / 4, (dt**3) / 2, (dt**2) / 2],
            [(dt**4) / 6, (dt**3) / 2, dt**2, dt],
            [(dt**3) / 6, (dt**2) / 2, dt, 1.0],
        ]

    if order_by_dim:
        return block_diag(*[Q] * block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var
