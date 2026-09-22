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


# ============================================================
# COM Octave oracle — Butterworth_Filter extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
# param.fb_BW_cutoff=0.75, param.fb=106.25e9, f=[0 10e9 40e9 79.6875e9 200e9]:
#
#   use_BW=1            -> [1+0j,
#                           0.94648161896813487-0.3227576349210437j,
#                           0.20240919992647205-0.97724929994732523j,
#                           -0.70710656237316283-0j,
#                           0.012169725890660104+0.022060200405580519j]
#   f 2x3, use_BW=0     -> ones(1,3), THREE elements: MATLAB length() is the
#                          LONGEST dimension, not the first one
#   f scalar, use_BW=0  -> 1          (length(scalar) is 1)
#   use_BW=[]           -> ones branch    ("if []"    is false)
#   use_BW=[1 0]        -> ones branch    ("if [1 0]" is false)
#   use_BW=[1 1]        -> filter branch  ("if [1 1]" is true)
# ============================================================

OCT_F = np.array([0.0, 10e9, 40e9, 79.6875e9, 200e9])
OCT_H = np.array([1 + 0j,
                  0.94648161896813487 - 0.3227576349210437j,
                  0.20240919992647205 - 0.97724929994732523j,
                  -0.70710656237316283 - 0j,
                  0.012169725890660104 + 0.022060200405580519j])


def oct_param():
    return make_param(cutoff=0.75, fb=106.25e9)


def test_oracle_values():
    """Pin the COM Octave response for a real COM axis."""
    H = Butterworth_Filter(oct_param(), OCT_F, 1)
    np.testing.assert_allclose(H, OCT_H, rtol=1e-13, atol=0)


def test_length_is_longest_dimension_not_first():
    """ones(1,length(f)) for a 2x3 f is THREE ones, not two."""
    f = np.array([[0.0, 1e9, 2e9], [3e9, 4e9, 5e9]])
    H = Butterworth_Filter(oct_param(), f, 0)
    assert H.size == 3
    np.testing.assert_array_equal(H, np.ones(3))


def test_scalar_f_disabled_returns_one():
    """length() of a scalar is 1; len() raised TypeError on an unsized object."""
    H = Butterworth_Filter(oct_param(), 40e9, 0)
    assert np.asarray(H).size == 1
    assert float(np.asarray(H).ravel()[0]) == 1.0


def test_use_bw_empty_is_false():
    """MATLAB `if []` is false, so an empty flag takes the all-pass branch."""
    H = Butterworth_Filter(oct_param(), OCT_F, np.array([]))
    np.testing.assert_array_equal(H, np.ones(5))


def test_use_bw_all_nonzero_is_true():
    """MATLAB `if [1 1]` is true — every element non-zero."""
    H = Butterworth_Filter(oct_param(), OCT_F, np.array([1, 1]))
    np.testing.assert_allclose(H, OCT_H, rtol=1e-13, atol=0)


def test_use_bw_any_zero_is_false():
    """MATLAB `if [1 0]` is FALSE; Python list truthiness called it true."""
    for flag in (np.array([1, 0]), [1, 0], (1, 0)):
        H = Butterworth_Filter(oct_param(), OCT_F, flag)
        np.testing.assert_array_equal(H, np.ones(5), err_msg=repr(flag))
