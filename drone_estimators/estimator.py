"""Base class for all estimators."""

from __future__ import absolute_import, annotations, division, print_function

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


class Estimator(ABC):
    """Base class for estimator implementations."""

    def __init__(self, state_dim: int, input_dim: int, obs_dim: int, dt: float):
        """Initialize basic parameters.

        Args:
            state_dim: Dimensionality of the systems states, e.g., x of f(x,u)
            input_dim: Dimensionality of the input to the dynamics, e.g., u of f(x,u)
            obs_dim: Dimensionality of the observations, e.g., y
            dt: Time step between callings.
        """
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._input_dim = input_dim
        self._dt = dt
        self._state = np.zeros(self._state_dim)
        self._input = np.zeros(self._input_dim)

    def reset(self):
        """Reset the noise to its initial state."""
        self._state = np.zeros(self._state_dim)
        self._input = np.zeros(self._input_dim)

    @abstractmethod
    def step(self):
        """Increment the noise step for time dependent noise classes."""
        pass

    def set_input(self, u: Array):
        """Sets the input of the dynamical system. Assuming this class gets called multiple times between controller calls. We therefore store the input as a constant in the class.

        Args:
            u: Input to the dynamical system.
        """
        # TODO check for shape?
        self.data = self.data.replace(u=u)
