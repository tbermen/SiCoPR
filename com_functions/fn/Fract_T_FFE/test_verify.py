"""Verification tests for Fract_T_FFE().

# ============================================================
# MATLAB GROUND TRUTH
# V0 = (circshift(V, skew_step) + V) / 2
#
# V=[1,2,3,4], skew_step=0:
#   V0 = (V + V)/2 = V = [1,2,3,4]
#
# V=[1,2,3,4], skew_step=1:
#   circshift([1,2,3,4],1) = [4,1,2,3]  (shift down by 1 → last wraps to front)
#   V0 = ([4,1,2,3]+[1,2,3,4])/2 = [2.5,1.5,2.5,3.5]
#
# V=[1,2,3,4], skew_step=-1:
#   circshift([1,2,3,4],-1) = [2,3,4,1]
#   V0 = ([2,3,4,1]+[1,2,3,4])/2 = [1.5,2.5,3.5,2.5]
#
# iscolumn: (N,1) input handled same as 1-D
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Fract_T_FFE.py_impl import Fract_T_FFE


def test_zero_skew_identity():
    """skew_step=0 → V0 = V."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(Fract_T_FFE(V, 0), V)


def test_positive_skew():
    """skew_step=1 → average with shift-right-by-1."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.array([2.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(Fract_T_FFE(V, 1), expected)


def test_negative_skew():
    """skew_step=-1 → average with shift-left-by-1."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.array([1.5, 2.5, 3.5, 2.5])
    np.testing.assert_allclose(Fract_T_FFE(V, -1), expected)


def test_column_vector_input():
    """(N,1) column vector → same result as 1-D."""
    V = np.array([[1.0], [2.0], [3.0], [4.0]])
    V0 = Fract_T_FFE(V, 0)
    np.testing.assert_allclose(V0, [1.0, 2.0, 3.0, 4.0])


def test_output_length():
    V = np.ones(16)
    assert len(Fract_T_FFE(V, 3)) == 16


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py on matlab/com_ieee8023_4p16p0.m).
#
# 1. circshift(V',[ishift,0])' shifts the TRANSPOSE along its rows, i.e. a 2-D
#    V is shifted across its COLUMNS. np.roll(V, skew_step) flattened V and
#    rolled the flat buffer, which mixes rows together.
# 2. A fractional skew_step is an error in MATLAB; np.roll truncated it and
#    answered.
# ============================================================

def test_2d_input_shifts_across_columns():
    """COM Octave: Fract_T_FFE([1 2;3 4;5 6], 1) -> [1.5 1.5; 3.5 3.5; 5.5 5.5].
    np.roll on the flat buffer gave [[3.5,1.5],[2.5,3.5],[4.5,5.5]]."""
    V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    V0 = Fract_T_FFE(V, 1)
    assert V0.shape == (3, 2)
    np.testing.assert_array_equal(V0, [[1.5, 1.5], [3.5, 3.5], [5.5, 5.5]])


def test_2d_input_wide():
    """COM Octave: Fract_T_FFE([1 2 3;4 5 6], 1) -> [2 1.5 2.5; 5 4.5 5.5]
    and skew -1 -> [1.5 2.5 2; 4.5 5.5 5]."""
    V = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(Fract_T_FFE(V, 1),
                                  [[2.0, 1.5, 2.5], [5.0, 4.5, 5.5]])
    np.testing.assert_array_equal(Fract_T_FFE(V, -1),
                                  [[1.5, 2.5, 2.0], [4.5, 5.5, 5.0]])


def test_fractional_skew_raises():
    """COM Octave: Fract_T_FFE([1 2 3 4], 1.5) ->
    'circshift: all values of N must be integers'.  np.roll truncated the
    1.5 to 1 and returned [2.5 1.5 2.5 3.5]."""
    with pytest.raises(ValueError):
        Fract_T_FFE(np.array([1.0, 2.0, 3.0, 4.0]), 1.5)


def test_1d_skew_wraps_past_end():
    """COM Octave: Fract_T_FFE([1 2 3 4 5 6], 3) -> [2.5 3.5 4.5 2.5 3.5 4.5]."""
    V = np.arange(1.0, 7.0)
    np.testing.assert_array_equal(Fract_T_FFE(V, 3),
                                  [2.5, 3.5, 4.5, 2.5, 3.5, 4.5])
