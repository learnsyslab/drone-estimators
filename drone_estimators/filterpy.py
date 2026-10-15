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

# import jax
import numpy as np
from scipy.linalg import block_diag

from drone_estimators.structs.estimator_data import EstimatorData
from drone_estimators.utils.batching import select
from drone_estimators.utils.integration import integrate_EstimatorData

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later
    from drone_estimators.structs.ukf_data import UKFSettings


def ukf_predict_correct(
    data: EstimatorData, settings: UKFSettings, mask: Array | None = None
) -> EstimatorData:
    """TODO. Only batch elements where mask is True are updated."""
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
    # SI = xp.linalg.inv(data.S)
    # data = data.replace(S=S, SI=SI)
    # data = data.replace(S=S)

    # compute cross variance
    Pxz = ukf_cross_variance(
        EstimatorData.as_state_array(data), zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc
    )
    # K = xp.dot(Pxz, data.SI)       # Kalman gain
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.mT, Pxz.mT).mT
    y = xp.subtract(data.z, zp)  # residual
    # data = data.replace(K=K, y=y)

    # Update Gaussian state estimate (x, P)
    x = EstimatorData.as_state_array(data) + (K @ y[..., None])[..., 0]
    # print(f"P prior = {xp.diag(P)}")
    # print(f"P prior = \n{P}")
    # Added identity for numerical stability
    P = data.covariance - K @ S @ K.mT  # + xp.eye(P.shape[0]) * 1e-9
    # print(f"P post = {xp.diag(P)}")
    # print(f"P post = \n{P}")

    # Save posterior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ukf_predict(
    data: EstimatorData, settings: UKFSettings, mask: Array | None = None
) -> EstimatorData:
    """TODO. Only batch elements where mask is True are updated."""
    # xp = data.pos.__array_namespace__()
    data_in = data
    #### Predict
    # Calculate sigma pointstf.transform.rotation.x,
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
    # print(f"{pos_dot=}")
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
    """TODO. Only batch elements where mask is True are updated."""
    xp = data.covariance.__array_namespace__()
    data_in = data
    # Pass mean and covariance of prediction through unscented transform
    zp, S = ukf_unscented_transform(
        data.sigmas_h, settings.SPsettings.Wm, settings.SPsettings.Wc, settings.R
    )
    # SI = xp.linalg.inv(data.S)
    # data = data.replace(S=S, SI=SI)
    # data = data.replace(S=S)

    # compute cross variance
    Pxz = ukf_cross_variance(
        EstimatorData.as_state_array(data), zp, data.sigmas_f, data.sigmas_h, settings.SPsettings.Wc
    )
    # K = xp.dot(Pxz, data.SI)       # Kalman gain
    # K @ S = Pxz => K = Pxz @ S^-1 => or: S.T @ K.T = Pxz.T
    K = xp.linalg.solve(S.mT, Pxz.mT).mT
    y = xp.subtract(data.z, zp)  # residual
    # data = data.replace(K=K, y=y)

    # Update Gaussian state estimate (x, P)
    x = EstimatorData.as_state_array(data) + (K @ y[..., None])[..., 0]
    # print(f"P prior = {xp.diag(P)}")
    # print(f"P prior = \n{P}")
    # Added identity for numerical stability
    P = data.covariance - K @ S @ K.mT  # + xp.eye(P.shape[0]) * 1e-9
    # print(f"P post = {xp.diag(P)}")
    # print(f"P post = \n{P}")

    # Save posterior
    data = EstimatorData.from_state_array(data, x)
    data = data.replace(covariance=P)

    return data if mask is None else select(mask, data, data_in)


def ukf_calculate_sigma_points(data: EstimatorData, settings: UKFSettings) -> Array:
    """TODO."""
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
    """TODO."""
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
    """Compute cross variance of the state `x` and measurement `z`."""
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
    """TODO."""
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
    """TODO."""
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
