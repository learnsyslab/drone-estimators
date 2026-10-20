import importlib.util
import subprocess
import sys

import pytest

needs_ros = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs ROS 2")


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "drone_estimators.estimator_kalman",
        "drone_estimators.estimator_diff",
        pytest.param("drone_estimators.ros.node", marks=needs_ros),
    ],
)
def test_import_first(module: str):
    """Tests that the estimators can be imported in a fresh interpreter before scipy is imported."""
    # -P keeps the working directory off sys.path, so the installed package is tested rather
    # than a source tree in the current directory
    subprocess.run([sys.executable, "-P", "-c", f"import {module}"], check=True)
