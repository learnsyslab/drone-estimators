"""Integration tests of the Kalman filter with crazyflow's dynamics."""

from __future__ import annotations

import numpy as np
import pytest
from crazyflow.dynamics import available_dynamics, dynamics_features, supported_drones

from drone_estimators.estimator_kalman import KalmanFilter
from drone_estimators.utils.dynamics import dynamics_function

# All dynamics with every drone they have parameters for
DYNAMICS_DRONES = [(d, drone) for d in available_dynamics for drone in supported_drones(d)]


# TODO test if the whole filterpy chain is jitable


@pytest.mark.integration
@pytest.mark.parametrize("dynamics, drone", DYNAMICS_DRONES)
def test_kalman(dynamics: str, drone: str):
    """Tests if the Kalman filter can be imported and stepped."""
    supports_dynamics = dynamics_features(dynamics_function(dynamics, drone))["rotor_dynamics"]
    kf = KalmanFilter(1 / 200, dynamics, drone, estimate_rotor_vel=supports_dynamics)

    kf.predict(1 / 240, np.array([0.0, 0.0, 0.0, 0.5]))

    kf.correct(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
