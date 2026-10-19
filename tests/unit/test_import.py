import subprocess
import sys

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "drone_estimators.estimator_kalman",
        "drone_estimators.estimator_diff",
        "drone_estimators.ros.node",
    ],
)
def test_import_first(module: str):
    """Tests that the estimators can be imported in a fresh interpreter before scipy is imported."""
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)
