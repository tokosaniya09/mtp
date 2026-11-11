"""Tests for the [Eng] filtered finite-difference differentiator (D15): src/vsfc/differentiator.py."""

import numpy as np
import pytest

from vsfc.differentiator import FilteredDifferentiator


def test_raw_difference_when_tau_zero():
    d = FilteredDifferentiator(0.1, 0.0)
    d.reset([0.0, 1.0])
    np.testing.assert_allclose(d.update([0.3, 0.8]), [3.0, -2.0])


def test_ramp_slope_recovered_after_transient():
    dt, tau, slope = 1e-3, 0.02, np.array([0.5, -1.0, 2.0])
    d = FilteredDifferentiator(dt, tau)
    d.reset(np.zeros(3))
    for k in range(1, 1001):                    # 1 s = 50 time constants
        y = d.update(slope * k * dt)
    np.testing.assert_allclose(y, slope, rtol=1e-12)


def test_first_order_step_response():
    dt, tau = 1e-3, 0.05
    d = FilteredDifferentiator(dt, tau)
    d.reset(np.zeros(1))
    y = [d.update([k * dt])[0] for k in range(1, int(tau / dt) + 1)]    # unit-slope ramp for one tau
    assert y[-1] == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_requires_reset_and_valid_args():
    with pytest.raises(RuntimeError):
        FilteredDifferentiator(0.1, 0.0).update([1.0])
    with pytest.raises(ValueError):
        FilteredDifferentiator(0.0, 0.1)
    with pytest.raises(ValueError):
        FilteredDifferentiator(0.1, -1.0)
