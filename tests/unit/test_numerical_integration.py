"""Unit tests of the numerical integration."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from crazyflow.sim.integration import _integrate as crazyflow_integrate
from scipy.spatial.transform import Rotation as R

from drone_estimators.utils.integration import _integrate


@pytest.mark.unit
@pytest.mark.parametrize("rotor_vel", [True, False])
def test_integration(rotor_vel: bool):
    """Tests the integration against crazyflow's implementation."""
    rng = np.random.default_rng(0)
    n = 41  # e.g. the sigma points of the UKF
    pos, vel, ang_vel, dvel, dang_vel = (rng.normal(size=(n, 3)) for _ in range(5))
    quat = R.random(n, rng=rng).as_quat()
    rotor, drotor = rng.uniform(1e4, 2e4, (n, 4)), rng.normal(0, 1e3, (n, 4))
    ang_vel[0] = [1e-310, 0.0, 0.0]  # Subnormal rotation, which is set to zero
    dt = 1 / 300
    with jax.enable_x64(True):
        expected = crazyflow_integrate(
            pos, quat, vel, ang_vel, rotor, vel, ang_vel, dvel, dang_vel, drotor, dt
        )
    if not rotor_vel:
        rotor, drotor = None, None
    actual = _integrate(pos, quat, vel, ang_vel, rotor, vel, ang_vel, dvel, dang_vel, drotor, dt)
    for x, x_expected in zip(actual[:4], expected[:4]):
        assert np.allclose(x, x_expected, rtol=1e-12, atol=1e-12)
    if rotor_vel:
        assert np.allclose(actual[4], expected[4], rtol=1e-12, atol=1e-12)
    else:
        assert actual[4] is None
