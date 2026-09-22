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


# ---------------------------------------------------------------------------
# Values below came from running ttos() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
# ---------------------------------------------------------------------------

def _T_nominal():
    T = np.zeros((2, 2, 2), dtype=complex)
    T[0, 0, :] = [1.25 + 0.1j, 1.4 - 0.05j]
    T[0, 1, :] = [-0.15 + 0.02j, -0.08 + 0.01j]
    T[1, 0, :] = [0.12 - 0.03j, 0.07 + 0.02j]
    T[1, 1, :] = [0.9 + 0.01j, 0.95 - 0.02j]
    return T


def test_oracle_nominal_2x2x2():
    """COM Octave, first slice of ttos()."""
    s = ttos(_T_nominal())
    assert s.shape == (2, 2, 2)
    np.testing.assert_allclose(
        s[:, :, 0].ravel(order='F'),
        [0.09348171701112877 - 0.031478537360890298j,
         0.79491255961844198 - 0.063593004769475353j,
         0.91339268680445174 + 0.0034085850556438684j,
         0.1179650238473768 - 0.025437201907790145j])


def test_oracle_delta_uses_the_original_t11_not_eps():
    """MATLAB forms delta BEFORE substituting eps for a zero t11.

    The mirror of the stot case: with t22 = 0 the determinant is -t21*t12,
    unaffected by t11, but s12 = delta/t11 divides by eps either way. What
    separates the two orderings is t22 non-zero and t11 zero, where delta is
    t11*t22 - t21*t12 = -t21*t12 exactly only if the original t11 is used.
    """
    T = np.zeros((2, 2, 1), dtype=complex)
    T[0, 0, 0] = 0.0
    T[0, 1, 0] = 0.25
    T[1, 0, 0] = 0.5
    T[1, 1, 0] = 4.0
    s = ttos(T)
    # delta = 0*4 - 0.5*0.25 = -0.125 exactly; s12 = delta/eps
    np.testing.assert_allclose(s[0, 1, 0].real,
                               -0.125 / np.finfo(float).eps, rtol=1e-15)


def test_oracle_zero_t11_divides_by_eps():
    """COM Octave: t11 = 0 becomes eps, so s21 = 1/eps = 4503599627370496."""
    T = _T_nominal()
    T[0, 0, 0] = 0.0
    s = ttos(T)
    np.testing.assert_allclose(s[1, 0, 0].real, 4503599627370496.0, rtol=0, atol=0)


def test_oracle_roundtrip_stot_of_ttos_returns_T():
    """COM Octave: stot(ttos(T)) reproduces T to rounding."""
    from com_functions.fn.stot.py_impl import stot
    T = _T_nominal()
    np.testing.assert_allclose(stot(ttos(T)), T, rtol=1e-12, atol=1e-14)


def test_oracle_input_is_not_mutated():
    """MATLAB passes by value, so t11(t11==0)=eps cannot reach the caller."""
    T = _T_nominal()
    T[0, 0, 0] = 0.0
    ttos(T)
    assert T[0, 0, 0] == 0.0
