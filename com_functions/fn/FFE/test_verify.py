"""Verification tests for FFE().

# ============================================================
# MATLAB GROUND TRUTH
# ishift = (i-1-cmx)*spui  (MATLAB 1-based i)
#        = (i_py - cmx)*spui  (Python 0-based i_py = i-1)
#
# C=[1.0], cmx=0, spui=1, V=[1,2,3,4]:
#   i=0: ishift=0 → circshift(V,0)=[1,2,3,4] → V0=[1,2,3,4]
#
# C=[0.5,1.0], cmx=1, spui=1, V=[1,2,3,4]:
#   i=0 (c=0.5): ishift=(0-1)*1=-1 → roll([1,2,3,4],-1)=[2,3,4,1] → 0.5*[2,3,4,1]
#   i=1 (c=1.0): ishift=0 → roll([1,2,3,4],0)=[1,2,3,4]
#   V0 = [1,1.5,2,0.5] + [1,2,3,4] = [2,3.5,5,4.5]
#
# iscolumn: (N,1) input → same result as 1-D input
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.FFE.py_impl import FFE


def test_single_tap_zero_shift():
    """C=[1.0], cmx=0: identity → V0=V."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    V0 = FFE(np.array([1.0]), cmx=0, spui=1, V=V)
    np.testing.assert_allclose(V0, V)


def test_two_taps_cursor_at_index1():
    """C=[0.5,1.0], cmx=1, spui=1: ishift=-1 and 0."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    V0 = FFE(np.array([0.5, 1.0]), cmx=1, spui=1, V=V)
    np.testing.assert_allclose(V0, [2.0, 3.5, 5.0, 4.5])


def test_zero_tap_skipped():
    """C=[0.0,1.0]: only cursor tap contributes → V0=V."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    V0 = FFE(np.array([0.0, 1.0]), cmx=1, spui=1, V=V)
    np.testing.assert_allclose(V0, V)


def test_column_vector_input():
    """(N,1) column vector input treated same as 1-D."""
    V = np.array([[1.0], [2.0], [3.0], [4.0]])   # shape (4,1)
    V0 = FFE(np.array([1.0]), cmx=0, spui=1, V=V)
    np.testing.assert_allclose(V0, [1.0, 2.0, 3.0, 4.0])


def test_spui_multiplies_shift():
    """spui=2 doubles the shift index."""
    V = np.array([1.0, 2.0, 3.0, 4.0])
    # C=[1.0,0.0,0.5], cmx=0, spui=2: only i=0 (shift=0) and i=2 (shift=(2-0)*2=4=0 mod 4)
    V0 = FFE(np.array([1.0, 0.0, 0.5]), cmx=0, spui=2, V=V)
    # i=0: ishift=0 → [1,2,3,4]*1.0; i=2: ishift=4 → roll([1,2,3,4],4)=[1,2,3,4]*0.5
    expected = V * 1.0 + np.roll(V, 4) * 0.5   # roll by 4 = identity for len-4
    np.testing.assert_allclose(V0, expected)


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py, driving the body from matlab/com_ieee8023_4p16p0.m
# -- the octave_compat copy of FFE is a rewritten speed variant).
#
# 1. circshift(V',[ishift,0]) shifts the TRANSPOSE along its rows. For a 2-D V
#    that shifts across columns and returns an M-by-N result; np.roll(V,ishift)
#    flattened V and rolled the flat buffer, giving the wrong values AND the
#    wrong shape.
# 2. A fractional ishift (a non-integer spui) is an error in MATLAB;
#    np.roll truncated it and answered.
# ============================================================

def test_2d_input_shifts_the_transpose():
    """COM Octave: FFE([0.5 1], 1, 1, [1 2;3 4;5 6]) -> 2x3
    [2 5 8; 2.5 5.5 8.5].  np.roll on the flat buffer gave 3x2
    [[2,3.5],[5,6.5],[8,6.5]]."""
    V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    V0 = FFE(np.array([0.5, 1.0]), cmx=1, spui=1, V=V)
    assert V0.shape == (2, 3)
    np.testing.assert_array_equal(V0, [[2.0, 5.0, 8.0], [2.5, 5.5, 8.5]])


def test_2d_input_wide():
    """COM Octave: FFE([1 0 2], 0, 1, [1 2 3;4 5 6]) -> 3x2
    [5 14; 8 17; 5 14]."""
    V = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    V0 = FFE(np.array([1.0, 0.0, 2.0]), cmx=0, spui=1, V=V)
    assert V0.shape == (3, 2)
    np.testing.assert_array_equal(V0, [[5.0, 14.0], [8.0, 17.0], [5.0, 14.0]])


def test_fractional_shift_raises():
    """COM Octave: FFE([1 1], 0, 1.5, [1 2 3 4]) ->
    'circshift: all values of N must be integers'.  np.roll truncated the
    1.5 to 1 and returned [5 3 5 7]."""
    with pytest.raises(ValueError):
        FFE(np.array([1.0, 1.0]), cmx=0, spui=1.5, V=np.array([1.0, 2.0, 3.0, 4.0]))


def test_fractional_cmx_that_yields_whole_shift_is_allowed():
    """COM Octave: FFE([1 1], 0.5, 2, [1 2 3 4]) -> [6;4;6;4].
    Only the product (i-1-cmx)*spui has to be whole, not cmx itself."""
    V0 = FFE(np.array([1.0, 1.0]), cmx=0.5, spui=2, V=np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(V0, [6.0, 4.0, 6.0, 4.0])


def test_matches_ffe_fast_on_the_same_shifts():
    """FFE and FFE_Fast must agree when V_shift holds the same circshifts.

    COM Octave, C=[0.1 0 -0.3 0.7], cmx=2, spui=3, V=1:12, both give
    [7.4 7.9 8.4 0.5 1 1.5 0.8 1.3 1.8 2.3 2.8 3.3].
    """
    from com_functions.fn.FFE_Fast.py_impl import FFE_Fast
    C = np.array([0.1, 0.0, -0.3, 0.7])
    V = np.arange(1.0, 13.0)
    cmx, spui = 2, 3
    V_shift = np.column_stack([np.roll(V, (k - cmx) * spui) for k in range(len(C))])
    expected = [7.4000000000000004, 7.8999999999999995, 8.3999999999999986,
                0.5, 1, 1.5, 0.79999999999999982, 1.3000000000000003,
                1.7999999999999998, 2.2999999999999994, 2.7999999999999998,
                3.3000000000000003]
    np.testing.assert_array_equal(FFE(C, cmx, spui, V), expected)
    np.testing.assert_array_equal(FFE_Fast(C, V_shift), expected)
