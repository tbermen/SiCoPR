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


# ---------------------------------------------------------------------------
# Values below came from running dfe_clipper() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
# ---------------------------------------------------------------------------

def test_oracle_row_and_column_keep_their_orientation():
    """COM Octave: [1 5 3] clipped by [2 4 3.5]/[0 1 2] -> [1 4 3].

    A row in gives 1x3 out, a column gives 3x1 -- dfe_clipper's isrow() branch
    exists to align the thresholds, and the result must follow the input.
    """
    x, hi, lo = (np.array([1., 5, 3]), np.array([2., 4, 3.5]),
                 np.array([0., 1, 2]))
    np.testing.assert_allclose(dfe_clipper(x, hi, lo), [1.0, 4.0, 3.0])
    col = dfe_clipper(x.reshape(-1, 1), hi.reshape(-1, 1), lo.reshape(-1, 1))
    np.testing.assert_allclose(col.ravel(), [1.0, 4.0, 3.0])
    assert col.shape == (3, 1)


def test_oracle_row_input_with_column_thresholds():
    """COM Octave: row input, column thresholds -> 1x3 [1 4 3]."""
    out = dfe_clipper(np.array([1., 5, 3]),
                      np.array([[2.], [4.], [3.5]]),
                      np.array([[0.], [1.], [2.]]))
    np.testing.assert_allclose(out.ravel(), [1.0, 4.0, 3.0])


def test_oracle_nan_passes_through_unclipped():
    """COM Octave: dfe_clipper([1 NaN 3],[2 2 2],[0 0 0]) -> [1 NaN 2].

    Both comparisons are false for NaN, so it is neither raised nor lowered.
    """
    out = dfe_clipper(np.array([1., np.nan, 3.]), np.full(3, 2.0), np.zeros(3))
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [1.0, 2.0])


def test_oracle_crossed_bounds_use_the_original_input():
    """COM Octave: dfe_clipper([0 1.5 3],[1 1 1],[2 2 2]) -> [2 2 1].

    min > max. The second mask tests the ORIGINAL input, not the already
    max-clipped value, so element 2 is lowered to 1 and then raised to 2.
    Recomputing the mask from the running output would give [2 2 2].
    """
    np.testing.assert_allclose(
        dfe_clipper(np.array([0., 1.5, 3.]), np.ones(3), np.full(3, 2.0)),
        [2.0, 2.0, 1.0])


def test_oracle_scalar_threshold_legal_only_within_bounds():
    """MATLAB indexes the THRESHOLD with the input-shaped logical mask.

    COM Octave: dfe_clipper([3 1 1], 2, -9) -> [2 1 1]   (only position 1 true)
                dfe_clipper([1 3 1], 2, -9) -> error: max_threshold(2): out of
                                               bound 1
    numpy would broadcast the scalar in both cases and return an answer for a
    call the reference refuses.
    """
    np.testing.assert_allclose(
        dfe_clipper(np.array([3., 1, 1]), np.array(2.0), np.array(-9.0)),
        [2.0, 1.0, 1.0])
    with pytest.raises(IndexError):
        dfe_clipper(np.array([1., 3, 1]), np.array(2.0), np.array(-9.0))
