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


# ============================================================
# COM Octave oracle values — tools/octave_oracle.py runs synth_tline verbatim
# out of octave/com_ieee8023_4p16p0_octave_compat.m (which is byte-identical to
# the body in matlab/com_ieee8023_4p16p0.m, and to matlab_source.m here).
# Literals pinned 2026-09-22.
#
# Two divergences these pin:
#  * f < 0.  MATLAB's sqrt()/log() of a negative real go complex; numpy's on a
#    float array return NaN, so the port used to answer NaN where the reference
#    answers a finite complex number.
#  * Z_c == -2*Z_0.  rho_rl divides by zero: MATLAB carries Inf through to an
#    all-NaN result, the port raised ZeroDivisionError.
#
# Last-bit note: with rho_rl != 0 the two differ by <= 1.1e-16 because Octave
# and numpy round complex multiplication differently (Octave's z.^2 == z.*z and
# numpy's z**2 == z*z, but the two products disagree on 6 of 33 sampled
# points).  That is a library/FMA difference, not arithmetic form, so the
# rho_rl != 0 case is pinned to 1e-15 relative rather than bit-exactly.
# ============================================================

def test_octave_negative_frequency():
    """MATLAB goes complex for f<0; the float path returned NaN."""
    f = np.array([-53e9, -1e9, -1.0, 0.0, 1.0, 1e9, 53e9])
    gc = [0.0002, 0.0034, 0.0001]
    s11, s12, s21, s22 = synth_tline(f, 100.0, 50.0, gc, 1e-11, 0.15)
    assert np.all(np.isfinite(s21)), "f<0 must not produce NaN"
    exp_s21 = np.array([
        1.006069830981237 - 0.0057570668335386373j,
        1.0005250077183339 - 0.00051026778874684064j,
        0.99997001657717277 - 1.6126934619039383e-08j,
        0.99997000044999551 - 0j,
        0.99996998432284823 - 1.6127329872222888e-08j,
        0.99944502400616742 - 0.00050971701585516053j,
        0.99547098018189539 - 0.0016957280280588414j])
    np.testing.assert_array_equal(s21, exp_s21)      # bit-for-bit here
    np.testing.assert_allclose(s11, 0.0, atol=0.0)


def test_octave_degenerate_rho_denominator():
    """Z_c == -2*Z_0 divides by zero: MATLAB returns all-NaN, not an error."""
    f = np.array([0.0, 1e9, 10e9, 26.5625e9, 53e9])
    with np.errstate(invalid="ignore", divide="ignore"):
        s11, s12, s21, s22 = synth_tline(f, -100.0, 50.0,
                                         [0.0002, 0.0034, 0.0001], 1e-11, 0.15)
    assert np.all(np.isnan(s11)), "expected MATLAB's all-NaN, got %r" % (s11,)
    assert np.all(np.isnan(s21)), "expected MATLAB's all-NaN, got %r" % (s21,)


def test_octave_nominal_reflective_line():
    """Z_c=130, Z_0=50, d=0.5 — the full s11/s21 expression with rho_rl != 0."""
    f = np.array([0.0, 1e9, 10e9, 26.5625e9, 53e9])
    s11, s12, s21, s22 = synth_tline(f, 130.0, 50.0,
                                     [0.0005, 0.005, 0.0002], 1.3e-11, 0.5)
    exp_s11 = np.array([
        6.6328996214314688e-05 + 0j,
        0.0007558261169519372 + 0.00065955776655119238j,
        0.0024180794451299976 + 0.0016769154999624427j,
        0.0041372725306026642 + 0.0018850046491843235j,
        0.0061566883801420528 + 0.0012138384629711001j])
    exp_s21 = np.array([
        0.99974138180182581 + 0j,
        0.99705242169086361 - 0.0025784209955412433j,
        0.99055162861809531 - 0.0065958048362888813j,
        0.98377360842051575 - 0.0074620699317742758j,
        0.97573334207161233 - 0.0048422426399998561j])
    # s11 loses relative precision to the 1-exp_gd**2 cancellation, so it is
    # pinned absolutely against the O(1) intermediates rather than relatively.
    np.testing.assert_allclose(s11, exp_s11, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(s21, exp_s21, rtol=1e-15, atol=0)


def test_octave_zero_Zc_shorted_line():
    """Z_c=0 with d>0 -> rho_rl=-1: full reflection, no transmission."""
    f = np.array([0.0, 1e9, 10e9, 26.5625e9, 53e9])
    s11, s12, s21, s22 = synth_tline(f, 0.0, 50.0,
                                     [0.0002, 0.0034, 0.0001], 1e-11, 0.15)
    np.testing.assert_array_equal(s11, np.full(5, -1.0, dtype=complex))
    np.testing.assert_array_equal(s21, np.zeros(5, dtype=complex))
