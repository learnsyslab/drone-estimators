"""Unit tests of the wrapper around crazyflow's dynamics."""

from __future__ import annotations

import pytest
from crazyflow.dynamics import available_dynamics, supported_drones

from drone_estimators.utils.dynamics import dynamics_function

# All dynamics with every drone they have parameters for
DYNAMICS_DRONES = [(d, drone) for d in available_dynamics for drone in supported_drones(d)]


@pytest.mark.unit
@pytest.mark.parametrize("dynamics, drone", DYNAMICS_DRONES)
def test_model_loading(dynamics: str, drone: str):
    """Tests if the models for the kalman filters can be imported."""
    dynamics_function(dynamics, drone)
