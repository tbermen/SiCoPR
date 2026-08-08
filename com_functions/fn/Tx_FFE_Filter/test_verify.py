"""Verification tests for Tx_FFE_Filter().

# ============================================================
# MATLAB GROUND TRUTH
# H_TxFFE = sum_i( Tx_FFE(i) * exp(-j2π*(i-icur)*f/fb) )
# where icur = argmax(Tx_FFE), 1-based in MATLAB.
# (ii_matlab - icur_matlab) == (ii_python - icur_python) → formula identical.
#
# Use_Tx_FFE=0 (or False):   H = ones(1, length(f))
#
# Single tap [1.0], Use_Tx_FFE=1:
#   icur=1 (MATLAB) / 0 (Python); only term: 1.0*exp(0) = 1 for all f
#   H = all ones
#
# Two taps [0.5, 1.0], icur=2 (MATLAB) / 1 (Python):
#   f=0:    H = 0.5*exp(0) + 1.0*exp(0) = 1.5
#   f=fb/2: H = 0.5*exp(jπ) + 1.0 = 0.5*(-1) + 1.0 = 0.5
#   f=fb:   H = 0.5*exp(j2π) + 1.0 = 0.5 + 1.0 = 1.5
#
# Three taps [0.1, 1.0, 0.2]:
#   f=0:    H = 0.1 + 1.0 + 0.2 = 1.3
#   General: H(0) = sum(taps)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Tx_FFE_Filter.py_impl import Tx_FFE_Filter


def make_param(fb=25e9, taps=None):
    p = SimpleNamespace(fb=fb)
    if taps is not None:
        p.Pkg_TXFFE_preset = taps
    return p


def test_disabled_returns_ones():
    """Use_Tx_FFE=0 → H = all ones."""
    f = np.array([0.0, 5e9, 10e9, 25e9])
    H = Tx_FFE_Filter(make_param(), f, Use_Tx_FFE=0)
    np.testing.assert_array_equal(H, np.ones(4))


def test_single_tap_is_unity():
    """Single tap [1.0] → H = all ones (exp(0) = 1)."""
    f = np.array([0.0, 5e9, 12.5e9, 25e9])
    H = Tx_FFE_Filter(make_param(taps=[1.0]), f, Use_Tx_FFE=1)
    np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-14)


def test_two_taps_dc_value():
    """[0.5, 1.0] taps: H(0) = sum([0.5, 1.0]) = 1.5."""
    H = Tx_FFE_Filter(make_param(fb=25e9, taps=[0.5, 1.0]),
                      np.array([0.0]), Use_Tx_FFE=1)
    assert H[0].real == pytest.approx(1.5, rel=1e-12)
    assert abs(H[0].imag) < 1e-14


def test_two_taps_half_baud():
    """[0.5, 1.0] taps at f=fb/2: H = 0.5*exp(jπ)+1 = 0.5."""
    fb = 25e9
    H = Tx_FFE_Filter(make_param(fb=fb, taps=[0.5, 1.0]),
                      np.array([fb / 2]), Use_Tx_FFE=1)
    assert H[0].real == pytest.approx(0.5, rel=1e-12)
    assert abs(H[0].imag) < 1e-10


def test_dc_equals_sum_of_taps():
    """H(f=0) = sum(taps) for any tap vector."""
    taps = [0.1, 1.0, -0.05, 0.2]
    H = Tx_FFE_Filter(make_param(fb=25e9, taps=taps),
                      np.array([0.0]), Use_Tx_FFE=1)
    assert H[0].real == pytest.approx(sum(taps), rel=1e-12)


def test_output_length_matches_input():
    f = np.linspace(0, 25e9, 30)
    H = Tx_FFE_Filter(make_param(taps=[0.2, 1.0, -0.1]), f, Use_Tx_FFE=1)
    assert len(H) == 30
