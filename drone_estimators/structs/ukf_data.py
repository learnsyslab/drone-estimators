"""Settings of the unscented Kalman filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flax.struct import Callable, dataclass

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


@dataclass
class UKFSettings:
    """Settings of the unscented Kalman filter."""

    SPsettings: SigmaPointsSettings
    Q: Array
    R: Array
    fx: Callable[
        [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
        tuple[Array, Array, Array, Array, Array | None],
    ]
    hx: Callable[[Array, Array, Array, Array, Array, Array, Array | None, Array | None], Array]

    @classmethod
    def create(
        cls,
        SPsettings: SigmaPointsSettings,
        Q: Array,
        R: Array,
        fx: Callable[
            [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
            tuple[Array, Array, Array, Array, Array | None],
        ],
        hx: Callable[[Array, Array, Array, Array, Array, Array, Array | None, Array | None], Array],
    ) -> UKFSettings:
        """Create the settings from the sigma point settings, noise covariances, and the model.

        Args:
            SPsettings: The sigma point settings.
            Q: The process noise covariance.
            R: The measurement noise covariance.
            fx: The continuous dynamics function.
            hx: The observation function.

        Returns:
            The settings.
        """
        return cls(SPsettings, Q, R, fx, hx)


@dataclass
class SigmaPointsSettings:
    """Parameters and weights of the Van der Merwe sigma points."""

    n: int
    alpha: float
    beta: float
    kappa: float
    lambda_: float
    Wc: Array
    Wm: Array

    @classmethod
    def create(cls, n: int, alpha: float, beta: float, kappa: float = 0.0) -> SigmaPointsSettings:
        """Compute the weights from the scaling parameters.

        Args:
            n: The dimension of the state.
            alpha: The spread of the sigma points around the mean.
            beta: Prior knowledge of the distribution, 2 is optimal for Gaussians.
            kappa: Secondary scaling parameter.

        Returns:
            The settings with the weights for the mean and the covariance.
        """
        lambda_ = alpha**2 * (n + kappa) - n
        c = 0.5 / (n + lambda_)
        Wc0 = np.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)])
        Wm0 = np.array([lambda_ / (n + lambda_)])
        Wc = np.full(2 * n, c)
        Wm = np.full(2 * n, c)
        Wc = np.concat((Wc0, Wc))
        Wm = np.concat((Wm0, Wm))

        return cls(n, alpha, beta, kappa, lambda_, Wc, Wm)
