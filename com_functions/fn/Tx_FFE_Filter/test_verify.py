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


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, 4p16p0 compat file,
# Tx_FFE_Filter + varargin_extractor).  Generated 2026-09-22.
# ============================================================


def test_empty_preset_returns_zeros():
    """Pkg_TXFFE_preset=[] : MATLAB max([]) gives an empty icur and `for ii=1:0`
    runs zero times, so H stays at zeros(1,length(f)).

    COM Octave, param.fb=25e9, param.Pkg_TXFFE_preset=[], f=[0 12.5e9 25e9],
    Use_Tx_FFE=1  ->  H_TxFFE == [0 0 0].
    np.argmax on an empty array raised ValueError instead.
    """
    f = np.array([0.0, 12.5e9, 25e9])
    H = Tx_FFE_Filter(make_param(fb=25e9, taps=[]), f, Use_Tx_FFE=1)
    assert len(H) == 3
    np.testing.assert_array_equal(H, np.zeros(3))


def test_default_f_axis_stops_at_or_below_fb():
    """f omitted : MATLAB f=0:10e6:param.fb never passes fb.

    COM Octave, param.Pkg_TXFFE_preset=[0.1 1.0 -0.2], Use_Tx_FFE=1, f=[]:
        fb=53.125e9   -> numel(H_TxFFE) = 5313
        fb=106.25e9   -> numel(H_TxFFE) = 10626
        fb=26.5625e9  -> numel(H_TxFFE) = 2657
    np.arange(0, fb+10e6, 10e6) overshot by one point whenever fb was not a
    multiple of 10e6 (5314 / 2658).
    """
    for fb, n in ((53.125e9, 5313), (106.25e9, 10626), (26.5625e9, 2657)):
        H = Tx_FFE_Filter(make_param(fb=fb, taps=[0.1, 1.0, -0.2]), None,
                          Use_Tx_FFE=1)
        assert len(H) == n, 'fb=%g' % fb


def test_default_f_axis_last_point():
    """The last default-f sample is floor(fb/10e6)*10e6, not the next step up.

    COM Octave, param.fb=53.125e9, Pkg_TXFFE_preset=[0.1 1.0 -0.2], f=[]:
        H_TxFFE(end) = 0.90000001748525005 - 0.00017740757480374955i
    """
    H = Tx_FFE_Filter(make_param(fb=53.125e9, taps=[0.1, 1.0, -0.2]), None,
                      Use_Tx_FFE=1)
    assert H[-1].real == pytest.approx(0.90000001748525005, rel=1e-13)
    assert H[-1].imag == pytest.approx(-0.00017740757480374955, rel=1e-11)
