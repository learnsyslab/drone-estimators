"""Integration tests of saving the estimates in the ROS node."""

from __future__ import annotations

from collections import defaultdict

import jax
import numpy as np
import pytest

from drone_estimators.estimator_kalman import KalmanFilter
from drone_estimators.estimator_legacy import StateEstimator
from drone_estimators.ros_nodes.ros2_utils import append_state

LEGACY_PARAMS = (0.0001, 0.007, 0.09, 0.005, 0.07)  # As in the ROS node


@pytest.mark.integration
@pytest.mark.parametrize("estimator_type", ["legacy", "ukf"])
def test_append_state(estimator_type: str):
    """Tests if the estimates of all estimators can be saved."""
    pos, quat = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])
    with jax.enable_x64(True):
        if estimator_type == "legacy":
            estimator = StateEstimator(LEGACY_PARAMS)
        else:
            estimator = KalmanFilter(
                1 / 200,
                "so_rpy_rotor_drag",
                "cf21B_500",
                estimate_rotor_vel=True,
                estimate_dist_f=True,
            )
        estimator.set_state(pos, quat)
        data = defaultdict(list)
        for _ in range(3):
            append_state(data, 0.0, estimator.predict(1 / 200))
            append_state(data, 0.0, estimator.correct(pos, quat))
    assert len(data["pos"]) == len(data["covariance"]) == 6
    expected = 0 if estimator_type == "legacy" else 6
    assert len(data["forces_motor"]) == len(data["forces_dist"]) == expected
    assert len(data["torques_dist"]) == 0
