"""Verification tests for vref_intersect() — linear interpolation of eye contour.

# ============================================================
# MATLAB GROUND TRUTH
# m1 = y(x_in) - y(x_in-1)
# b1 = y(x_in) - m1*x_in
# line_intersection = (vref - b1) / m1
# All indices 1-based; result is fractional 1-based index.
#
# ec = [[0.1],[0.3],[0.7],[0.9]]  (4 rows, column 1)
#
# Case 1: x_in=3, vref=0.5
#   y_curr=0.7, y_prev=0.3, m1=0.4
#   b1 = 0.7 - 0.4*3 = -0.5
#   result = (0.5 - (-0.5)) / 0.4 = 2.5
#
# Case 2: x_in=3, vref=0  (typical COM usage, vref=0)
#   b1 = -0.5
#   result = (0 - (-0.5)) / 0.4 = 1.25
#
# Case 3: exact crossing (y_curr == vref)
#   ec=[[0.1],[0.5],[0.9]], x_in=2, vref=0.5
#   y_curr=0.5, y_prev=0.1, m1=0.4
#   b1 = 0.5 - 0.4*2 = -0.3
#   result = (0.5 - (-0.3)) / 0.4 = 2.0  (exactly at x_in)
#
# Case 4: negative slope
#   ec=[[0.9],[0.5],[0.1]], x_in=2, vref=0.3
#   m1 = 0.5-0.9 = -0.4
#   b1 = 0.5 - (-0.4)*2 = 1.3
#   result = (0.3 - 1.3) / (-0.4) = 2.5
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.vref_intersect.py_impl import vref_intersect


def _ec(values):
    """Build a single-column eye_contour array."""
    return np.array(values, dtype=float).reshape(-1, 1)


def test_rising_crossing_mid():
    ec = _ec([0.1, 0.3, 0.7, 0.9])
    assert vref_intersect(ec, 3, 0.5) == pytest.approx(2.5)


def test_rising_crossing_zero_vref():
    ec = _ec([0.1, 0.3, 0.7, 0.9])
    assert vref_intersect(ec, 3, 0.0) == pytest.approx(1.25)


def test_exact_crossing_at_x_in():
    """When y_curr == vref the result should be exactly x_in."""
    ec = _ec([0.1, 0.5, 0.9])
    assert vref_intersect(ec, 2, 0.5) == pytest.approx(2.0)


def test_negative_slope():
    ec = _ec([0.9, 0.5, 0.1])
    assert vref_intersect(ec, 2, 0.3) == pytest.approx(2.5)


def test_returns_scalar():
    ec = _ec([0.1, 0.3, 0.7, 0.9])
    result = vref_intersect(ec, 3, 0.5)
    assert np.isscalar(result)


def test_result_between_brackets():
    """For a monotone rising contour, result must lie between x_in-1 and x_in."""
    ec = _ec([0.0, 0.2, 0.8, 1.0])
    result = vref_intersect(ec, 3, 0.5)
    assert 2.0 <= result <= 3.0
