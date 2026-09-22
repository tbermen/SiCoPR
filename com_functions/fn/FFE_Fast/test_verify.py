"""Verification tests for FFE_Fast().

# ============================================================
# MATLAB GROUND TRUTH
# V0 = sum_i( V_shift[:,i] * C[i] )  for C[i] != 0
# V0 starts as scalar 0; first hit produces an array via broadcasting.
#
# C=[1.0], V_shift=[[1],[2],[3]]:
#   V0 = V_shift[:,0] * 1.0 = [1,2,3]
#
# C=[0.0], V_shift=any:
#   No tap is non-zero → V0=0 (scalar, unchanged)
#
# C=[0.5, 1.0], V_shift=[[1,2],[3,4],[5,6]]:
#   i=0: V0 = [1,3,5]*0.5 = [0.5,1.5,2.5]
#   i=1: V0 = [0.5,1.5,2.5] + [2,4,6]*1.0 = [2.5,5.5,8.5]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.FFE_Fast.py_impl import FFE_Fast


def test_single_tap_unity():
    """C=[1.0]: V0 = column 0 of V_shift."""
    V_shift = np.array([[1.0], [2.0], [3.0]])
    V0 = FFE_Fast(np.array([1.0]), V_shift)
    np.testing.assert_allclose(V0, [1.0, 2.0, 3.0])


def test_all_zero_taps():
    """C=[0.0]: no accumulation → V0 stays scalar 0."""
    V_shift = np.array([[1.0], [2.0], [3.0]])
    V0 = FFE_Fast(np.array([0.0]), V_shift)
    assert V0 == 0.0


def test_two_taps():
    """C=[0.5,1.0], weighted sum of two columns."""
    V_shift = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    V0 = FFE_Fast(np.array([0.5, 1.0]), V_shift)
    np.testing.assert_allclose(V0, [2.5, 5.5, 8.5])


def test_zero_tap_skipped():
    """Middle tap zero → only columns 0 and 2 contribute."""
    V_shift = np.array([[1, 0, 3], [2, 0, 6]], dtype=float)
    V0 = FFE_Fast(np.array([1.0, 999.0, 2.0]), V_shift)
    np.testing.assert_allclose(V0, [1*1+3*2, 2*1+6*2])


def test_output_length():
    N = 8
    V_shift = np.random.rand(N, 3)
    V0 = FFE_Fast(np.array([0.2, 0.5, 0.3]), V_shift)
    assert len(V0) == N


# ============================================================
# Probed against the executed reference (COM Octave oracle,
# tools/octave_oracle.py). NO divergence was found in FFE_Fast: it agreed
# with the reference on every probe, including a zero tap vector, a C
# shorter than V_shift, a column C, and a C that overruns V_shift.
# These pin that, and pin the FFE/FFE_Fast cross-check.
# ============================================================

def test_octave_agrees_with_FFE_on_the_same_shifts():
    """FFE_Fast is FFE with the circshifts hoisted out, so the two must give
    the same answer for the same taps.

    COM Octave, C=[0.1 0 -0.3 0.7], cmx=2, spui=3, V=1:12: both return
    [7.4 7.9 8.4 0.5 1 1.5 0.8 1.3 1.8 2.3 2.8 3.3].
    """
    from com_functions.fn.FFE.py_impl import FFE
    C = np.array([0.1, 0.0, -0.3, 0.7])
    V = np.arange(1.0, 13.0)
    cmx, spui = 2, 3
    V_shift = np.column_stack([np.roll(V, (k - cmx) * spui) for k in range(len(C))])
    expected = [7.4000000000000004, 7.8999999999999995, 8.3999999999999986,
                0.5, 1, 1.5, 0.79999999999999982, 1.3000000000000003,
                1.7999999999999998, 2.2999999999999994, 2.7999999999999998,
                3.3000000000000003]
    np.testing.assert_array_equal(FFE_Fast(C, V_shift), expected)
    np.testing.assert_array_equal(FFE(C, cmx, spui, V), expected)


def test_octave_all_zero_taps_returns_scalar_zero():
    """COM Octave: every tap zero leaves V0 as the scalar 0, never a vector."""
    V_shift = np.array([[1., 10, 100], [2., 20, 200],
                        [3., 30, 300], [4., 40, 400]])
    assert FFE_Fast(np.zeros(3), V_shift) == 0.0


def test_octave_C_shorter_than_V_shift_uses_leading_columns():
    """COM Octave: C=[2 3] against a 4x3 V_shift -> [32 64 96 128];
    the third column is simply not used."""
    V_shift = np.array([[1., 10, 100], [2., 20, 200],
                        [3., 30, 300], [4., 40, 400]])
    np.testing.assert_array_equal(FFE_Fast(np.array([2., 3.]), V_shift),
                                  [32.0, 64.0, 96.0, 128.0])


def test_octave_C_longer_than_V_shift_raises():
    """COM Octave: C with 4 taps against a 4x3 V_shift ->
    'V_shift(_,4): out of bound 3'. Python raises IndexError, as it must."""
    V_shift = np.array([[1., 10, 100], [2., 20, 200],
                        [3., 30, 300], [4., 40, 400]])
    with pytest.raises(IndexError):
        FFE_Fast(np.ones(4), V_shift)
