"""Utilities for batched estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import jax
from array_api_compat import array_namespace

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later

T = TypeVar("T")


def select(mask: Array, new: T, old: T) -> T:
    """Take the batch elements of new where mask is True and of old otherwise.

    Args:
        mask: Boolean array with the batch shape of the data, e.g., (n_drones,).
        new: Pytree of arrays with leading batch dimensions.
        old: Pytree with the same structure and shapes as new.
    """

    def _select(a: Array, b: Array) -> Array:
        xp = array_namespace(mask, a, b)
        return xp.where(xp.reshape(mask, mask.shape + (1,) * (a.ndim - mask.ndim)), a, b)

    return jax.tree.map(_select, new, old)
