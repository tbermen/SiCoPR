"""Verification tests for Butterworth_Filter().

# ============================================================
# MATLAB GROUND TRUTH
# 4th-order Butterworth: H = 1/polyval([1 2.613126 3.414214 2.613126 1], j*f/fc)
# where fc = param.fb_BW_cutoff * param.fb
#
# use_BW=False → H = ones(1,N) for any f
#
# use_BW=True, f=0 (DC):
#   polyval(p, 0) = p(end) = 1  →  H_dc = 1+0j
#
# use_BW=True, f=fc (normalized cutoff, s=j):
#   p(j) = j^4 + 2.613126*j^3 + 3.414214*j^2 + 2.613126*j + 1
#        = 1 - 2.613126j - 3.414214 + 2.613126j + 1
#        = -1.414214 + 0j
#   H = 1/(-1.414214) = -1/sqrt(2)  →  |H| = 1/sqrt(2) ≈ 0.70711  (-3 dB)
#
# use_BW=True, f=[0, fc] → H = [1+0j, -1/sqrt(2)+0j]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Butterworth_Filter.py_impl import Butterworth_Filter


def make_param(cutoff=1.0, fb=1.0):
    return SimpleNamespace(fb_BW_cutoff=cutoff, fb=fb)


def test_passthrough_when_disabled():
    """use_BW=False → all ones regardless of frequency."""
    f = np.array([0.0, 1e9, 10e9])
    H = Butterworth_Filter(make_param(), f, use_BW=False)
    np.testing.assert_array_equal(H, np.ones(3))


def test_dc_unity():
    """use_BW=True, f=0 → H = 1 (DC gain = 1)."""
    param = make_param(cutoff=1.0, fb=1.0)
    H = Butterworth_Filter(param, np.array([0.0]), use_BW=True)
    assert H[0] == pytest.approx(1.0)


def test_cutoff_minus3dB():
    """|H| = 1/sqrt(2) at f = fb_BW_cutoff * fb (normalized cutoff)."""
    param = make_param(cutoff=0.5, fb=2.0)   # fc = 1.0 Hz
    H = Butterworth_Filter(param, np.array([1.0]), use_BW=True)
    assert abs(H[0]) == pytest.approx(1.0 / np.sqrt(2), rel=1e-6)


def test_cutoff_value_exact():
    """At s=j the polynomial evaluates to exactly -sqrt(2), so H = -1/sqrt(2)."""
    param = make_param(cutoff=1.0, fb=1.0)
    H = Butterworth_Filter(param, np.array([1.0]), use_BW=True)
    expected = -1.0 / np.sqrt(2)
    assert H[0].real == pytest.approx(expected, rel=1e-6)
    assert abs(H[0].imag) < 1e-10


def test_output_length_matches_input():
    f = np.linspace(0, 5e9, 20)
    H = Butterworth_Filter(make_param(cutoff=1.0, fb=25e9), f, use_BW=True)
    assert len(H) == 20


def test_high_frequency_rolls_off():
    """Response decreases with frequency above cutoff."""
    param = make_param(cutoff=1.0, fb=1.0)
    f = np.array([0.5, 1.0, 2.0, 5.0])
    H = Butterworth_Filter(param, f, use_BW=True)
    magnitudes = np.abs(H)
    assert magnitudes[0] > magnitudes[1] > magnitudes[2] > magnitudes[3]
