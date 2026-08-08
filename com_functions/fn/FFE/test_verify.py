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
