"""Settings of the unscented Kalman filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flax.struct import Callable, dataclass

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


@dataclass
class UKFSettings:
    """TODO."""

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
        """TODO."""
        return cls(SPsettings, Q, R, fx, hx)


@dataclass
class SigmaPointsSettings:
    """TODO."""

    n: int
    alpha: float
    beta: float
    kappa: float
    lambda_: float
    Wc: Array
    Wm: Array

    @classmethod
    def create(cls, n: int, alpha: float, beta: float, kappa: float = 0.0) -> SigmaPointsSettings:
        """TODO."""
        lambda_ = alpha**2 * (n + kappa) - n
        c = 0.5 / (n + lambda_)
        Wc0 = np.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)])
        Wm0 = np.array([lambda_ / (n + lambda_)])
        Wc = np.full(2 * n, c)
        Wm = np.full(2 * n, c)
        Wc = np.concat((Wc0, Wc))
        Wm = np.concat((Wm0, Wm))

        return cls(n, alpha, beta, kappa, lambda_, Wc, Wm)
