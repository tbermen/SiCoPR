"""Verification tests for stot() — S-to-T parameter conversion.

# ============================================================
# MATLAB GROUND TRUTH
# t11=1/s21, t12=-s22/s21, t21=s11/s21, t22=-delta/s21
# delta = s11*s22 - s12*s21
#
# Case 1: s=[[0,0],[1,0]] (thru: s21=1, all others 0)
#   delta=0, t=[[1,0],[0,0]]
#
# Case 2: s=[[0.1, 0.9],[0.9, 0.1]]
#   s11=0.1, s12=0.9, s21=0.9, s22=0.1
#   delta=0.1*0.1-0.9*0.9=0.01-0.81=-0.8
#   t11=1/0.9≈1.1111, t12=-0.1/0.9≈-0.1111
#   t21=0.1/0.9≈0.1111, t22=-(-0.8)/0.9≈0.8889
#
# Case 3: multi-frequency (2,2,2) — result has same shape
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.stot.py_impl import stot


def test_thru_network():
    s = np.array([[0+0j, 0+0j], [1+0j, 0+0j]])
    t = stot(s)
    np.testing.assert_allclose(t, [[1, 0], [0, 0]], atol=1e-12)


def test_known_values():
    s = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=complex)
    t = stot(s)
    assert t[0, 0] == pytest.approx(1.0/0.9, rel=1e-10)
    assert t[0, 1] == pytest.approx(-0.1/0.9, rel=1e-10)
    assert t[1, 0] == pytest.approx(0.1/0.9, rel=1e-10)
    assert t[1, 1] == pytest.approx(0.8/0.9, rel=1e-10)


def test_output_shape_2d():
    s = np.eye(2, dtype=complex)
    assert stot(s).shape == (2, 2)


def test_output_shape_3d():
    s = np.zeros((2, 2, 5), dtype=complex)
    s[1, 0, :] = 1.0   # s21=1 for all freqs
    assert stot(s).shape == (2, 2, 5)


def test_zero_s21_guard():
    """s21==0 is replaced by eps — no ZeroDivisionError."""
    s = np.zeros((2, 2), dtype=complex)
    t = stot(s)   # should not raise
    assert np.all(np.isfinite(t))


def test_roundtrip_with_ttos():
    """stot followed by ttos should recover the original S-matrix."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from com_functions.fn.ttos.py_impl import ttos
    rng = np.random.default_rng(42)
    s_orig = rng.standard_normal((2, 2, 10)) + 1j*rng.standard_normal((2, 2, 10))
    s_orig[1, 0, :] += 0.5   # ensure s21 != 0
    s_recovered = ttos(stot(s_orig))
    np.testing.assert_allclose(s_recovered, s_orig, rtol=1e-10)
