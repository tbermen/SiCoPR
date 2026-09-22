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


# ---------------------------------------------------------------------------
# Values below came from running stot() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
# ---------------------------------------------------------------------------

def _S_nominal():
    S = np.zeros((2, 2, 2), dtype=complex)
    S[0, 0, :] = [0.1 + 0.02j, 0.05 - 0.01j]
    S[0, 1, :] = [0.8 - 0.1j, 0.7 - 0.2j]
    S[1, 0, :] = [0.8 - 0.1j, 0.7 - 0.2j]
    S[1, 1, :] = [0.12 + 0.03j, 0.06 - 0.02j]
    return S


def test_oracle_nominal_2x2x2():
    """COM Octave, first frequency slice of stot() on a reciprocal 2-port."""
    t = stot(_S_nominal())
    assert t.shape == (2, 2, 2)
    np.testing.assert_allclose(
        t[:, :, 0].ravel(order='F'),
        [1.2307692307692308 + 0.15384615384615385j,
         0.12000000000000001 + 0.040000000000000001j,
         -0.14307692307692307 - 0.055384615384615379j,
         0.78680000000000017 - 0.1084j])


def test_oracle_delta_uses_the_original_s21_not_eps():
    """MATLAB computes delta BEFORE substituting eps for a zero s21.

    With s11 = s22 = 0 the determinant is -s12*s21, which is exactly zero for
    the original s21 and -s12*eps if the substitution has already happened.
    t22 = -delta/s21 therefore comes out as 0 one way and as a whole s12 the
    other -- not a rounding difference.

    COM Octave: s12 = 0.7-0.2j, s21 = 0, rest 0  ->  t22 = -0
    """
    S = np.zeros((2, 2, 1), dtype=complex)
    S[0, 1, 0] = 0.7 - 0.2j
    t = stot(S)
    assert t[1, 1, 0] == 0, 'delta must be formed before s21 is replaced by eps'


def test_oracle_zero_s21_divides_by_eps():
    """COM Octave: a zero s21 becomes eps, so t11 = 1/eps = 4503599627370496."""
    S = _S_nominal()
    S[1, 0, 1] = 0.0
    t = stot(S)
    np.testing.assert_allclose(t[0, 0, 1].real, 4503599627370496.0, rtol=0, atol=0)
    np.testing.assert_allclose(t[1, 1, 1],
                               -12610078956637.389 + 7205759403792.793j, rtol=1e-12)


def test_oracle_plain_2x2_keeps_its_shape():
    """COM Octave returns a plain 2x2 for a 2x2 input, not 2x2x1."""
    t = stot(_S_nominal()[:, :, 0])
    assert t.shape == (2, 2)
    np.testing.assert_allclose(t.ravel(order='F')[0],
                               1.2307692307692308 + 0.15384615384615385j)


def test_oracle_input_is_not_mutated():
    """MATLAB passes by value, so s21(s21==0)=eps cannot reach the caller."""
    S = _S_nominal()
    S[1, 0, 1] = 0.0
    stot(S)
    assert S[1, 0, 1] == 0.0
