"""Verification tests for ttos() — T-to-S parameter conversion.

# ============================================================
# MATLAB GROUND TRUTH
# s11=t21/t11, s12=delta/t11, s21=1/t11, s22=-t12/t11
# delta = t11*t22 - t21*t12
#
# Case 1: t=[[1,0],[0,0]] (from thru S-matrix via stot)
#   t11=1, t12=0, t21=0, t22=0, delta=0
#   s=[[0,0],[1,0]]  → thru S-matrix recovered
#
# Case 2: known t-params from Case 2 of stot test
#   t11=1/0.9, t12=-0.1/0.9, t21=0.1/0.9, t22=0.8/0.9
#   should recover s=[[0.1,0.9],[0.9,0.1]]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.ttos.py_impl import ttos
from com_functions.fn.stot.py_impl import stot


def test_thru_roundtrip():
    """ttos(stot(thru)) should recover the thru S-matrix."""
    s = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
    np.testing.assert_allclose(ttos(stot(s)), s, atol=1e-12)


def test_known_values_roundtrip():
    s_orig = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=complex)
    np.testing.assert_allclose(ttos(stot(s_orig)), s_orig, rtol=1e-10)


def test_output_shape_2d():
    t = np.eye(2, dtype=complex)
    assert ttos(t).shape == (2, 2)


def test_output_shape_3d():
    t = np.zeros((2, 2, 7), dtype=complex)
    t[0, 0, :] = 1.0
    assert ttos(t).shape == (2, 2, 7)


def test_zero_t11_guard():
    """t11==0 replaced by eps — no ZeroDivisionError."""
    t = np.zeros((2, 2), dtype=complex)
    s = ttos(t)
    assert np.all(np.isfinite(s))


def test_multifreq_roundtrip():
    rng = np.random.default_rng(7)
    s_orig = rng.standard_normal((2, 2, 8)) + 1j*rng.standard_normal((2, 2, 8))
    s_orig[1, 0, :] += 0.5
    np.testing.assert_allclose(ttos(stot(s_orig)), s_orig, rtol=1e-10)
