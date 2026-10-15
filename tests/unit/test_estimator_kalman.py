"""Unit tests of the Kalman filter estimator."""

from __future__ import annotations

import pytest

from drone_estimators.estimator_kalman import KalmanFilter


@pytest.mark.unit
def test_kalman_filter_type():
    """Tests that only implemented filter types are accepted."""
    with pytest.raises(AssertionError):
        KalmanFilter(1 / 200, filter_type="EKF")
