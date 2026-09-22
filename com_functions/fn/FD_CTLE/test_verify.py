"""Verification tests for FD_CTLE().

# ============================================================
# MATLAB GROUND TRUTH
# hctf = (10^(kacdc_dB/20) + j*freq/f_z) / ((1+j*freq/f_p1)*(1+j*freq/f_p2))
#
# At DC (freq=0):
#   hctf = 10^(kacdc_dB/20) / 1  (purely real)
#   kacdc_dB=0   → H_dc = 1
#   kacdc_dB=20  → H_dc = 10
#   kacdc_dB=-20 → H_dc = 0.1
#
# kacdc_dB=0, f_z=f_p1=f_p2=1e9, freq=1e9:
#   num = 1 + j
#   den = (1+j)*(1+j) = 2j
#   H = (1+j)/(2j) = (1+j)*(-j)/2 = (1-j)/2  →  |H|=1/sqrt(2)
#
# kacdc_dB=0, f_z=1e9, f_p1=2e9, f_p2=3e9, freq=1e9:
#   num = 1+j
#   den = (1+0.5j)*(1+j/3) = 5/6 + 5j/6 = (5/6)*(1+j)
#   H = (1+j)/((5/6)*(1+j)) = 6/5 = 1.2  (real)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.FD_CTLE.py_impl import FD_CTLE


def test_dc_gain_zero_dB():
    """kacdc_dB=0, freq=0 → H=1."""
    H = FD_CTLE(np.array([0.0]), 1e9, 2e9, 4e9, kacdc_dB=0.0)
    assert H[0] == pytest.approx(1.0)


def test_dc_gain_positive_dB():
    """kacdc_dB=20, freq=0 → H=10."""
    H = FD_CTLE(np.array([0.0]), 1e9, 2e9, 4e9, kacdc_dB=20.0)
    assert H[0].real == pytest.approx(10.0, rel=1e-10)


def test_dc_gain_negative_dB():
    """kacdc_dB=-20, freq=0 → H=0.1."""
    H = FD_CTLE(np.array([0.0]), 1e9, 2e9, 4e9, kacdc_dB=-20.0)
    assert H[0].real == pytest.approx(0.1, rel=1e-10)


def test_equal_pole_zero():
    """kacdc_dB=0, f_z=f_p1=f_p2=1e9, freq=1e9 → H=(1-j)/2."""
    H = FD_CTLE(np.array([1e9]), 1e9, 1e9, 1e9, kacdc_dB=0.0)
    assert H[0].real == pytest.approx(0.5, rel=1e-10)
    assert H[0].imag == pytest.approx(-0.5, rel=1e-10)


def test_real_gain_case():
    """kacdc_dB=0, f_z=1e9, f_p1=2e9, f_p2=3e9, freq=1e9 → H=1.2 (real)."""
    H = FD_CTLE(np.array([1e9]), 1e9, 2e9, 3e9, kacdc_dB=0.0)
    assert H[0].real == pytest.approx(1.2, rel=1e-10)
    assert abs(H[0].imag) < 1e-10


def test_output_length_matches_input():
    freq = np.linspace(0, 50e9, 25)
    H = FD_CTLE(freq, 5e9, 15e9, 30e9, kacdc_dB=-3.0)
    assert len(H) == 25


# ============================================================
# COM Octave oracle — FD_CTLE extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
# No divergence was found: every probe below agrees to <= 3e-16 absolute, which
# is the last-bit difference between MATLAB's and numpy's complex division.
#
# freq = [0 1e9 20e9 53.125e9 120e9]
#   FD_CTLE(freq, 6e9, 15e9, 30e9, -5)  ->
#       [0.56234132519034907+0j,
#        0.57456348774080934+0.10945354799925074j,
#        1.6771109905437327-0.18799782978718932j,
#        0.78683301947844086-0.88664283233853802j,
#        0.2014184786598183-0.56719284697039296j]
#   FD_CTLE(freq, 0.66e9, 0.66e9, 100e100, -5)   (the engine's high-pass call)
#       [0.56234132519034907+0j,
#        0.86720248067213435+0.20120836261797825j,
#        0.99952390816713821+0.014427025238234043j,
#        0.99993246039748795+0.0054364263385733944j,
#        0.99998676122555985+0.0024070498981936597j]
#   FD_CTLE(20e9, 6e9, 15e9, 30e9, [-5 -6])      (10.^ is element-wise, so a
#       [1.6771109905437327-0.18799782978718932j,   vector gain is legal here
#        1.6754174926235246-0.15751486722344027j]   unlike S_RN's 10^)
#   FD_CTLE([], ...) -> empty
#   FD_CTLE([NaN Inf], 6e9, 15e9, 30e9, -5) -> [NaN+NaNj, NaN+NaNj]
# ============================================================

OCT_FREQ = np.array([0.0, 1e9, 20e9, 53.125e9, 120e9])


def test_oracle_values():
    H = FD_CTLE(OCT_FREQ, 6e9, 15e9, 30e9, -5.0)
    np.testing.assert_allclose(
        H, [0.56234132519034907 + 0j,
            0.57456348774080934 + 0.10945354799925074j,
            1.6771109905437327 - 0.18799782978718932j,
            0.78683301947844086 - 0.88664283233853802j,
            0.2014184786598183 - 0.56719284697039296j], rtol=1e-13, atol=0)


def test_oracle_highpass_call():
    """f_p1 = f_z and f_p2 = 100e100, as OptFom_Compute_CTLE calls it."""
    H = FD_CTLE(OCT_FREQ, 0.66e9, 0.66e9, 100e100, -5.0)
    np.testing.assert_allclose(
        H, [0.56234132519034907 + 0j,
            0.86720248067213435 + 0.20120836261797825j,
            0.99952390816713821 + 0.014427025238234043j,
            0.99993246039748795 + 0.0054364263385733944j,
            0.99998676122555985 + 0.0024070498981936597j], rtol=1e-13, atol=0)


def test_vector_gain_is_element_wise():
    """MATLAB writes 10.^(kacdc_dB/20) here, so a vector of gains is legal."""
    H = FD_CTLE(20e9, 6e9, 15e9, 30e9, np.array([-5.0, -6.0]))
    np.testing.assert_allclose(
        H, [1.6771109905437327 - 0.18799782978718932j,
            1.6754174926235246 - 0.15751486722344027j], rtol=1e-13, atol=0)


def test_empty_freq():
    assert FD_CTLE(np.array([]), 6e9, 15e9, 30e9, -5.0).size == 0


def test_nan_and_inf_freq():
    H = FD_CTLE(np.array([np.nan, np.inf]), 6e9, 15e9, 30e9, -5.0)
    assert np.all(np.isnan(H.real)) and np.all(np.isnan(H.imag))
