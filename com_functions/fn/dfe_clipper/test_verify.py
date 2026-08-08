"""Verification tests for dfe_clipper() — element-wise clip with per-element bounds.

# ============================================================
# MATLAB GROUND TRUTH
# Hand-computed from the formula:
#   clip_output = input
#   clip_output(input > max) = max(input > max)
#   clip_output(input < min) = min(input < min)
#
# Case 1 (nominal, mixed clipping):
#   input=[1,2,3,4,5], max=[3,3,3,3,3], min=[2,2,2,2,2]
#   1<2 → clip to 2; 2==2 → keep 2; 3==3 → keep 3; 4>3 → clip to 3; 5>3 → clip to 3
#   → [2, 2, 3, 3, 3]
#
# Case 2 (no clipping):
#   input=[2, 2.5, 3], max=[4,4,4], min=[1,1,1]  →  [2, 2.5, 3]
#
# Case 3 (all below min):
#   input=[-1,-2,-3], max=[2,2,2], min=[0,0,0]  →  [0, 0, 0]
#
# Case 4 (per-element different thresholds):
#   input=[1, 5, 3], max=[2, 4, 3.5], min=[0, 1, 2]
#   1 in [0,2] → 1; 5>4 → 4; 3 in [2,3.5] → 3   → [1, 4, 3]
#
# Case 5 (column vector 2-D input):
#   input=[[1],[5],[3]], max=[[2],[4],[3.5]], min=[[0],[1],[2]]
#   → [[1],[4],[3]]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.dfe_clipper.py_impl import dfe_clipper


def test_nominal_mixed_clipping():
    """Values below min clipped up, values above max clipped down."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hi = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    lo = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    expected = np.array([2.0, 2.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(dfe_clipper(x, hi, lo), expected)


def test_no_clipping():
    """All values within bounds — output equals input."""
    x = np.array([2.0, 2.5, 3.0])
    hi = np.array([4.0, 4.0, 4.0])
    lo = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(dfe_clipper(x, hi, lo), x)


def test_all_clipped_to_min():
    """All values below min threshold."""
    x = np.array([-1.0, -2.0, -3.0])
    hi = np.array([2.0, 2.0, 2.0])
    lo = np.array([0.0, 0.0, 0.0])
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(dfe_clipper(x, hi, lo), expected)


def test_per_element_thresholds():
    """Different threshold per element."""
    x = np.array([1.0, 5.0, 3.0])
    hi = np.array([2.0, 4.0, 3.5])
    lo = np.array([0.0, 1.0, 2.0])
    expected = np.array([1.0, 4.0, 3.0])
    np.testing.assert_array_equal(dfe_clipper(x, hi, lo), expected)


def test_column_vector_input():
    """2-D column vector input — thresholds reshaped to column, output is column."""
    x = np.array([[1.0], [5.0], [3.0]])
    hi = np.array([2.0, 4.0, 3.5])
    lo = np.array([0.0, 1.0, 2.0])
    result = dfe_clipper(x, hi, lo)
    expected = np.array([[1.0], [4.0], [3.0]])
    assert result.shape == (3, 1)
    np.testing.assert_array_equal(result, expected)


def test_output_shape_matches_input():
    """Output shape must equal input shape."""
    x = np.array([1.0, 2.0, 3.0])
    hi = np.array([2.0, 2.0, 2.0])
    lo = np.array([0.0, 0.0, 0.0])
    assert dfe_clipper(x, hi, lo).shape == x.shape


def test_at_boundary_no_clipping():
    """Values exactly at threshold are not clipped (strict inequalities)."""
    x = np.array([2.0, 4.0])
    hi = np.array([4.0, 4.0])
    lo = np.array([2.0, 2.0])
    np.testing.assert_array_equal(dfe_clipper(x, hi, lo), x)
