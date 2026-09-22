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


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py).
#
# MATLAB reads eye_contour(x_in-1,1), so x_in<=1 asks for subscript 0 and
# errors, and a fractional x_in is not a legal subscript either. Python
# turned subscript 0 into ec[-1,0] and quietly read the LAST row, and
# int(x_in) truncated a fractional index -- both answered where the
# reference refuses, off the far end of the eye.
# ============================================================

def test_x_in_one_raises_instead_of_wrapping():
    """COM Octave: vref_intersect([0.1;0.3;0.7;0.9], 1, 0.5) ->
    'eye_contour(0,_): subscripts must be either integers 1 to (2^63)-1 or
    logicals'.  Python returned 0.5, computed from the last row."""
    ec = np.array([[0.1], [0.3], [0.7], [0.9]])
    with pytest.raises(IndexError):
        vref_intersect(ec, 1, 0.5)


def test_x_in_zero_or_negative_raises():
    """Subscript 0 and below are illegal for the same reason."""
    ec = np.array([[0.1], [0.3], [0.7], [0.9]])
    for bad in (0, -1):
        with pytest.raises(IndexError):
            vref_intersect(ec, bad, 0.5)


def test_fractional_x_in_raises():
    """COM Octave: x_in=2.5 -> 'eye_contour(2.5,_): subscripts must be
    either integers 1 to (2^63)-1 or logicals'.  int(x_in) truncated to 2
    and Python answered 3.0."""
    ec = np.array([[0.1], [0.3], [0.7], [0.9]])
    with pytest.raises(IndexError):
        vref_intersect(ec, 2.5, 0.5)


def test_x_in_two_is_the_lowest_legal_index():
    """COM Octave: vref_intersect([0.1;0.3;0.7;0.9], 2, 0.5) -> 3.
    x_in=2 reads rows 1 and 2, which is legal."""
    ec = np.array([[0.1], [0.3], [0.7], [0.9]])
    assert vref_intersect(ec, 2, 0.5) == 3.0


def test_flat_segment_is_infinite():
    """COM Octave: a flat pair gives m1=0 and the intersection is Inf."""
    ec = np.array([[0.1], [0.3], [0.3], [0.9]])
    assert np.isinf(vref_intersect(ec, 3, 0.5))
