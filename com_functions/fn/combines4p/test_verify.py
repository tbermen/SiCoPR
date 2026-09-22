"""Verification tests for combines4p().

# ============================================================
# MATLAB GROUND TRUTH (lines 5327-5370)
# S-parameter cascade signal-flow graph:
#   N = 1 - s22_a * s11_b
#   s11out = s11_a + s12_a*s21_a*s11_b / N
#   s12out = s12_a * s12_b / N
#   s21out = s21_b * s21_a / N
#   s22out = s22_b + s12_b*s21_b*s22_a / N
#
# Identity network: s11=s22=0, s21=s12=1 → s21out=1, s11out=0
# All inputs squeezed to 1-D complex arrays before return.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.combines4p.py_impl import combines4p


def test_identity_identity():
    """Two matched-thru networks cascade to identity."""
    N = 5
    s0 = np.zeros(N)
    s1 = np.ones(N)
    s11, s12, s21, s22 = combines4p(s0, s1, s1, s0, s0, s1, s1, s0)
    np.testing.assert_allclose(s21, 1.0, atol=1e-12)
    np.testing.assert_allclose(s11, 0.0, atol=1e-12)


def test_output_is_1d():
    """Output arrays are 1-D regardless of input shape."""
    N = 4
    s = np.zeros((1, 1, N))
    s21_in = np.ones((1, 1, N))
    _, _, s21out, _ = combines4p(s, s21_in, s21_in, s, s, s21_in, s21_in, s)
    assert s21out.ndim == 1
    assert len(s21out) == N


def test_formula_nontrivial():
    """Verify the cascade formula against hand-computed values."""
    # Single frequency point
    a11, a12, a21, a22 = 0.1, 0.9, 0.9, 0.2
    b11, b12, b21, b22 = 0.3, 0.7, 0.7, 0.15
    N = 1 - a22 * b11
    exp_s11 = a11 + a12 * a21 * b11 / N
    exp_s21 = b21 * a21 / N
    s11o, _, s21o, _ = combines4p(
        np.array([a11]), np.array([a12]), np.array([a21]), np.array([a22]),
        np.array([b11]), np.array([b12]), np.array([b21]), np.array([b22]),
    )
    assert s11o[0] == pytest.approx(exp_s11)
    assert s21o[0] == pytest.approx(exp_s21)


def test_complex_inputs():
    """Complex S-parameters handled correctly."""
    a21 = np.array([0.8 + 0.1j])
    b21 = np.array([0.7 - 0.1j])
    z = np.array([0.0 + 0.0j])
    _, _, s21out, _ = combines4p(z, a21, a21, z, z, b21, b21, z)
    expected = (b21 * a21)[0]
    assert s21out[0] == pytest.approx(expected)


def test_output_length_matches_input():
    """Output arrays have the same length as input arrays."""
    N = 7
    rng = np.random.default_rng(0)
    args = [rng.random(N) + 0j for _ in range(8)]
    outs = combines4p(*args)
    for o in outs:
        assert len(o) == N


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, 4p16p0 compat file,
# combines4p).  Generated 2026-09-22.
#
# The inputs below are dyadic: every product is exact in binary64 and each
# N = 1 - s22_a*s11_b is an exact power of two, so the quotients are exact too.
# That matters because Octave and numpy do NOT agree bit-for-bit on ordinary
# complex multiplication -- over 5000 random pairs, z1.*z2 differed in 2205 of
# them by up to 2.22e-16 (real .*, ./ and exp are bit-identical).  On exact
# inputs that noise disappears, so these literals pin the arithmetic FORM:
# any reassociation of the products, or a swapped factor order, moves them.
# ============================================================

_A11 = np.array([0.5 + 0.25j, -1.0 + 0.5j, 0.75 - 0.5j, 2.0 + 1.0j])
_A12 = np.array([0.5 - 0.5j, 0.25 + 0.75j, -1.5 + 0.25j, 0.5 + 0.5j])
_A21 = np.array([0.25 + 1.0j, -0.5 - 0.25j, 0.5 + 0.5j, -0.25 + 0.75j])
_A22 = np.array([1 + 1j, 0.25 + 0.25j, 1 + 1j, 1 + 1j])
_B11 = np.array([1 - 1j, 1 - 1j, -0.5 + 0.5j, -1.5 + 1.5j])
_B12 = np.array([0.75 + 0.5j, -0.25 + 0.5j, 1.0 - 0.25j, 0.5 - 0.75j])
_B21 = np.array([-0.5 + 0.25j, 0.5 + 0.5j, 0.25 - 1.0j, 1.25 + 0.5j])
_B22 = np.array([0.25 - 0.75j, 1.5 + 0.25j, -0.75 + 1.0j, 0.5 + 0.25j])


def test_oracle_exact_dyadic_cascade():
    """COM Octave, combines4p on the dyadic inputs above:
        s11out = [-0.5+0.5i, -1.75-0.5i, 1.125-0.5625i, 2.09375+0.71875i]
        s12out = [-0.625+0.125i, -0.875-0.125i, -0.71875+0.3125i, 0.15625-0.03125i]
        s21out = [0.375+0.4375i, -0.25-0.75i, 0.3125-0.1875i, -0.171875+0.203125i]
        s22out = [0.6875-0.1875i, 1.25+0.125i, -0.21875+0.46875i, 0.921875+0.328125i]
    """
    s11, s12, s21, s22 = combines4p(_A11, _A12, _A21, _A22, _B11, _B12, _B21, _B22)
    np.testing.assert_array_equal(
        s11, np.array([-0.5 + 0.5j, -1.75 - 0.5j, 1.125 - 0.5625j,
                       2.09375 + 0.71875j]))
    np.testing.assert_array_equal(
        s12, np.array([-0.625 + 0.125j, -0.875 - 0.125j, -0.71875 + 0.3125j,
                       0.15625 - 0.03125j]))
    np.testing.assert_array_equal(
        s21, np.array([0.375 + 0.4375j, -0.25 - 0.75j, 0.3125 - 0.1875j,
                       -0.171875 + 0.203125j]))
    np.testing.assert_array_equal(
        s22, np.array([0.6875 - 0.1875j, 1.25 + 0.125j, -0.21875 + 0.46875j,
                       0.921875 + 0.328125j]))


def test_oracle_singular_denominator():
    """s22_a*s11_b == 1 exactly, so N == 0 and MATLAB divides by zero rather
    than raising.

    COM Octave, s22_a=0.5+0.5i, s11_b=1-1i (element 1):
        s11out(1) =  Inf - Infi
        s12out(1) =  Inf - Infi
        s21out(1) = -Inf - Infi
        s22out(1) = -Inf - Infi
    """
    a22 = np.array([0.5 + 0.5j])
    b11 = np.array([1 - 1j])
    with np.errstate(divide='ignore', invalid='ignore'):
        s11, s12, s21, s22 = combines4p(_A11[:1], _A12[:1], _A21[:1], a22,
                                        b11, _B12[:1], _B21[:1], _B22[:1])
    assert s11[0] == complex(np.inf, -np.inf)
    assert s12[0] == complex(np.inf, -np.inf)
    assert s21[0] == complex(-np.inf, -np.inf)
    assert s22[0] == complex(-np.inf, -np.inf)
