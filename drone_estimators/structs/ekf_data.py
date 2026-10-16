"""Settings of the extended Kalman filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax.struct import Callable, dataclass

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


@dataclass
class EKFSettings:
    """Settings of the extended Kalman filter.

    The state is propagated with the nonlinear dynamics fx, the covariance with their Jacobian A.
    """

    A: Array  # Jacobian of the continuous dynamics w.r.t. the state
    H: Array  # Jacobian of the observation function w.r.t. the state
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
        A: Array,
        H: Array,
        Q: Array,
        R: Array,
        fx: Callable[
            [Array, Array, Array, Array, Array, Array, Array | None, Array | None],
            tuple[Array, Array, Array, Array, Array | None],
        ],
        hx: Callable[[Array, Array, Array, Array, Array, Array, Array | None, Array | None], Array],
    ) -> EKFSettings:
        """Create the settings from the Jacobians, the noise covariances, and the model."""
        return cls(A, H, Q, R, fx, hx)
