"""Verification tests for synth_tline() — IEEE 802.3 93A transmission-line model.

# ============================================================
# MATLAB GROUND TRUTH
# gamma_coeff(1,2,3) → Python [0,1,2] (1-based → 0-based)
#
# Case 1: d=0 → rho_rl=0, s11=0, s21=1 for any impedance/freq
#
# Case 2: matched impedance (Z_c=2*Z_0) → rho_rl=0
#   s11=0, s21=exp(-d*gamma)
#   For lossless (gamma_coeff=[0,0,0], tau=0): gamma=0, s21=1
#
# Case 3: zero-loss lossless matched line (gc=[0,0,0], tau=0, d=1)
#   gamma=0, exp_gd=1, denom=1, s11=0, s21=1
#
# Case 4: purely resistive loss (gc=[alpha,0,0], d=1, matched)
#   gamma=alpha (real), s21=exp(-alpha)
#
# Case 5: symmetry s11=s22, s12=s21
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.synth_tline.py_impl import synth_tline


def test_zero_length():
    """d=0 → s11=0, s21=1 regardless of impedance."""
    f = np.array([0.0, 1e9, 10e9])
    s11, s12, s21, s22 = synth_tline(f, 75.0, 50.0, [0.1, 0.05, 0.02], 1e-11, 0)
    np.testing.assert_allclose(s11, 0.0, atol=1e-12)
    np.testing.assert_allclose(s21, 1.0, atol=1e-12)


def test_matched_lossless():
    """Matched (Z_c=2*Z_0) + lossless (gc=0, tau=0) → s11=0, s21=1."""
    f = np.array([1e9, 5e9])
    s11, s12, s21, s22 = synth_tline(f, 100.0, 50.0, [0.0, 0.0, 0.0], 0.0, 1.0)
    np.testing.assert_allclose(s11, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(s21), 1.0, atol=1e-12)


def test_dc_point_handled():
    """f=0 must not produce NaN (the gamma override handles log(0))."""
    f = np.array([0.0, 1e9])
    s11, _, s21, _ = synth_tline(f, 100.0, 50.0, [0.01, 0.0, 0.0], 0.0, 0.1)
    assert np.all(np.isfinite(s11))
    assert np.all(np.isfinite(s21))


def test_resistive_loss_matched():
    """Matched line with only DC loss: |s21| = exp(-alpha*d)."""
    alpha = 0.5
    d = 0.2
    f = np.array([1e9])
    _, _, s21, _ = synth_tline(f, 100.0, 50.0, [alpha, 0.0, 0.0], 0.0, d)
    assert abs(s21[0]) == pytest.approx(np.exp(-alpha * d), rel=1e-10)


def test_symmetry():
    f = np.linspace(1e8, 10e9, 5)
    s11, s12, s21, s22 = synth_tline(f, 100.0, 50.0, [0.1, 0.05, 0.01], 1e-11, 0.1)
    np.testing.assert_array_equal(s11, s22)
    np.testing.assert_array_equal(s12, s21)


def test_passive():
    """|s11|^2 + |s21|^2 ≤ 1 (passivity)."""
    f = np.linspace(1e8, 50e9, 30)
    s11, _, s21, _ = synth_tline(f, 100.0, 50.0, [0.05, 0.1, 0.02], 1e-11, 0.1)
    power = np.abs(s11)**2 + np.abs(s21)**2
    assert np.all(power <= 1.0 + 1e-10)
