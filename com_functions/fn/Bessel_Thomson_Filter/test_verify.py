"""Verification tests for Bessel_Thomson_Filter().

# ============================================================
# MATLAB GROUND TRUTH
# H_bt = a(1)/polyval(fliplr(a), j*f/(fb_BT_cutoff*fb))
# where a = bessel(BTorder)
#
# BTorder=1: a=[1,1], fliplr=[1,1], a(1)=1
#   H = 1/polyval([1,1], s) = 1/(s+1)
#   f=0 (s=0): H = 1
#   f=1, fc=1 (s=j): H = 1/(1+j) = (1-j)/2  →  |H| = 1/sqrt(2)
#
# BTorder=2: a=[3,3,1], fliplr=[1,3,3], a(1)=3
#   H = 3/polyval([1,3,3], s) = 3/(s^2+3s+3)
#   f=0 (s=0): H = 3/3 = 1
#   f=1, fc=1 (s=j): H = 3/(−1+3j+3) = 3/(2+3j) = (6−9j)/13
#
# use_BT=False → H = ones(1,N)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Bessel_Thomson_Filter.py_impl import Bessel_Thomson_Filter


def make_param(order=1, cutoff=1.0, fb=1.0):
    return SimpleNamespace(BTorder=order, fb_BT_cutoff=cutoff, fb=fb)


def test_passthrough_when_disabled():
    """use_BT=False → all ones."""
    f = np.array([0.0, 1e9, 10e9])
    H = Bessel_Thomson_Filter(make_param(), f, use_BT=False)
    np.testing.assert_array_equal(H, np.ones(3))


def test_dc_unity_order1():
    """BTorder=1, f=0 → H=1 (DC gain = 1)."""
    H = Bessel_Thomson_Filter(make_param(order=1), np.array([0.0]), use_BT=True)
    assert H[0] == pytest.approx(1.0)


def test_dc_unity_order2():
    """BTorder=2, f=0 → H=1."""
    H = Bessel_Thomson_Filter(make_param(order=2), np.array([0.0]), use_BT=True)
    assert H[0] == pytest.approx(1.0)


def test_order1_at_cutoff():
    """BTorder=1, f=fc (s=j): H=1/(1+j) = (1-j)/2, |H|=1/sqrt(2)."""
    H = Bessel_Thomson_Filter(make_param(order=1, cutoff=1.0, fb=1.0),
                               np.array([1.0]), use_BT=True)
    assert H[0].real == pytest.approx(0.5, rel=1e-10)
    assert H[0].imag == pytest.approx(-0.5, rel=1e-10)
    assert abs(H[0]) == pytest.approx(1.0 / np.sqrt(2), rel=1e-10)


def test_order2_at_cutoff():
    """BTorder=2, f=fc (s=j): H=3/(2+3j)=(6-9j)/13."""
    H = Bessel_Thomson_Filter(make_param(order=2, cutoff=1.0, fb=1.0),
                               np.array([1.0]), use_BT=True)
    expected = (6 - 9j) / 13
    assert H[0].real == pytest.approx(expected.real, rel=1e-10)
    assert H[0].imag == pytest.approx(expected.imag, rel=1e-10)


def test_output_length_matches_input():
    f = np.linspace(0, 10e9, 15)
    H = Bessel_Thomson_Filter(make_param(order=3, cutoff=1.0, fb=25e9),
                               f, use_BT=True)
    assert len(H) == 15
