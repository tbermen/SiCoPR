"""Verification tests for OptFom_Calc_Hr().

# ============================================================
# MATLAB GROUND TRUTH
# H_r = Butterworth(f) .* Bessel_Thomson(f) .* Raised_Cosine(f)
# Each filter returns all-ones when its flag is False.
#
# All disabled (OP.BW=False, OP.BT=False, OP.RC=False):
#   H_r = 1 .* 1 .* 1 = ones
#
# Only BW enabled, f=0: BW DC gain = 1 → H_r = 1
# Only BT enabled, f=0: BT DC gain = 1 → H_r = 1
# Only RC enabled, f < RC_Start: RC = 1 → H_r = 1
# Only RC enabled, f > RC_end:   RC = 0 → H_r = 0
#
# All enabled, f=0: 1 * 1 * 1 = 1 (all filters have unity DC gain)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Calc_Hr.py_impl import OptFom_Calc_Hr


def make_param():
    return SimpleNamespace(
        fb=25e9, fb_BW_cutoff=1.0,
        BTorder=2, fb_BT_cutoff=1.0,
        RC_Start=5e9, RC_end=20e9,
    )


def all_off():
    return SimpleNamespace(Bessel_Thomson=False, Butterworth=False, Raised_Cosine=False)


def test_all_disabled_returns_ones():
    f = np.array([0.0, 5e9, 12.5e9, 25e9])
    H = OptFom_Calc_Hr(f, make_param(), all_off())
    np.testing.assert_array_equal(H, np.ones(4))


def test_only_butterworth_dc_unity():
    """BW only, f=0 → H = 1."""
    OP = SimpleNamespace(Bessel_Thomson=False, Butterworth=True, Raised_Cosine=False)
    H = OptFom_Calc_Hr(np.array([0.0]), make_param(), OP)
    assert H[0].real == pytest.approx(1.0, rel=1e-10)


def test_only_bessel_thomson_dc_unity():
    """BT only, f=0 → H = 1."""
    OP = SimpleNamespace(Bessel_Thomson=True, Butterworth=False, Raised_Cosine=False)
    H = OptFom_Calc_Hr(np.array([0.0]), make_param(), OP)
    assert H[0].real == pytest.approx(1.0, rel=1e-10)


def test_only_raised_cosine_passband():
    """RC only, f < RC_Start → H = 1."""
    OP = SimpleNamespace(Bessel_Thomson=False, Butterworth=False, Raised_Cosine=True)
    H = OptFom_Calc_Hr(np.array([0.0, 2e9]), make_param(), OP)
    np.testing.assert_allclose(H, 1.0, atol=1e-14)


def test_only_raised_cosine_stopband():
    """RC only, f > RC_end → H = 0."""
    OP = SimpleNamespace(Bessel_Thomson=False, Butterworth=False, Raised_Cosine=True)
    H = OptFom_Calc_Hr(np.array([25e9]), make_param(), OP)
    assert H[0] == pytest.approx(0.0, abs=1e-14)


def test_all_enabled_dc_unity():
    """All enabled, f=0 → H = 1 (product of three DC-unity filters)."""
    OP = SimpleNamespace(Bessel_Thomson=True, Butterworth=True, Raised_Cosine=True)
    H = OptFom_Calc_Hr(np.array([0.0]), make_param(), OP)
    assert abs(H[0]) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-22): OptFom_Calc_Hr run verbatim under Octave from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m for this function and for the three filters it
# calls), param.fb=106.25e9, fb_BT_cutoff=fb_BW_cutoff=0.75, BTorder=4,
# RC_Start=20e9, RC_end=40e9.
#
# These pin the behaviour of the three filters AS CALLED FROM HERE. This
# directory used to carry private inlined copies of them, and those copies were
# the pre-oracle forms -- already proved wrong in their own directories but
# still live here, which is exactly the failure mode a private copy creates.
#   - Raised_Cosine_Filter/Tukey_Window CONCATENATES three counted pieces
#     (f<fr, fr<=f<=fb, f>fb); the inlined element-wise np.where answered a
#     different vector for any f that is not ascending, and for ties at the
#     band edges.
#   - MATLAB `if use_BT` is true only for a non-empty value whose elements are
#     ALL non-zero; the inlined `if not use_BT` raised on a numpy array and
#     took the filter branch for [1, 0].
# ---------------------------------------------------------------------------
def _oracle_param():
    return SimpleNamespace(fb=106.25e9, fb_BT_cutoff=0.75, BTorder=4,
                           fb_BW_cutoff=0.75, RC_Start=20e9, RC_end=40e9,
                           f_r=0.75)


def _rc_only():
    return SimpleNamespace(Bessel_Thomson=0, Butterworth=0, Raised_Cosine=1)


def test_octave_raised_cosine_unsorted_f():
    """COM Octave, f NOT ascending -> the window is grouped, not element-wise.

    f = [25e9, 0, 35e9, 15e9] -> [1, 1, 0.85355339059327373, 0.14644660940672616]
    (an element-wise np.where gives [0.8535..., 1, 0.1464..., 1] instead)."""
    f = np.array([25e9, 0.0, 35e9, 15e9])
    H = OptFom_Calc_Hr(f, _oracle_param(), _rc_only())
    np.testing.assert_array_equal(
        np.asarray(H).ravel(),
        np.array([1.0, 1.0, 0.85355339059327373, 0.14644660940672616]))


def test_octave_raised_cosine_band_edge_ties():
    """COM Octave, duplicated band edges.

    f = [20e9, 20e9, 40e9, 40e9, 0, 90e9] -> [1, 1, 1, 0, 0, 0]
    (element-wise np.where gives [1, 1, 0, 0, 1, 0])."""
    f = np.array([20e9, 20e9, 40e9, 40e9, 0.0, 90e9])
    H = OptFom_Calc_Hr(f, _oracle_param(), _rc_only())
    np.testing.assert_array_equal(np.asarray(H).ravel(),
                                  np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_octave_raised_cosine_ascending():
    """COM Octave, ascending f: [0 10 20 30 40 53.125] GHz ->
    [1, 1, 1, 0.49999999999999989, 0, 0]."""
    f = np.array([0.0, 10e9, 20e9, 30e9, 40e9, 53.125e9])
    H = OptFom_Calc_Hr(f, _oracle_param(), _rc_only())
    np.testing.assert_array_equal(
        np.asarray(H).ravel(),
        np.array([1.0, 1.0, 1.0, 0.49999999999999989, 0.0, 0.0]))


@pytest.mark.parametrize('flag', [np.array([1, 0]), np.array([]), [1, 0], []])
def test_octave_flag_that_is_not_all_nonzero_is_off(flag):
    """COM Octave: use_BT=[1 0] and use_BT=[] both take the ones branch.

    The inlined `if not use_BT` raised ValueError on the arrays and took the
    FILTER branch for the list [1, 0]."""
    f = np.array([0.0, 10e9, 20e9, 30e9, 40e9, 53.125e9])
    OP = SimpleNamespace(Bessel_Thomson=flag, Butterworth=0, Raised_Cosine=0)
    np.testing.assert_array_equal(np.asarray(OptFom_Calc_Hr(f, _oracle_param(), OP)).ravel(),
                                  np.ones(6))


def test_octave_bessel_thomson_values():
    """COM Octave, BT only, BTorder=4, fb_BT_cutoff=0.75, fb=106.25e9.

    Complex arithmetic, so this is pinned to a tolerance: numpy's complex
    multiply/divide uses SIMD FMA while Octave's does not, which moves shared
    results by ~1 ulp on roughly half of all elements (measured: 1758/4000 on
    complex a.*b, 1699/4000 on a./b, while real a.*b and a.*b-c.*d matched
    4000/4000)."""
    f = np.array([0.0, 10e9, 20e9, 30e9, 40e9, 53.125e9])
    OP = SimpleNamespace(Bessel_Thomson=1, Butterworth=0, Raised_Cosine=0)
    H = np.asarray(OptFom_Calc_Hr(f, _oracle_param(), OP)).ravel()
    expect = np.array([
        1 + 0j,
        0.99102081252659346 - 0.12502035081263993j,
        0.96431687282217626 - 0.24723782428759919j,
        0.92058182535403177 - 0.36392993353316289j,
        0.86094792288791533 - 0.4725322755018197j,
        0.76117227029930112 - 0.59892268189033548j])
    np.testing.assert_allclose(H, expect, rtol=1e-15, atol=0)
