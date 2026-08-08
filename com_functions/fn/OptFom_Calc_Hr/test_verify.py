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
