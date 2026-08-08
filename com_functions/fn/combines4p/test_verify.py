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
