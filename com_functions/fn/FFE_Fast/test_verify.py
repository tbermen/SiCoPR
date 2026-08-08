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
