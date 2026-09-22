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


# ---------------------------------------------------------------------------
# Values below came from running Bessel_Thomson_Filter() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py. The same two traps Butterworth_Filter had; this
# function is structurally identical in the reference and was missed when that
# one was fixed on 2026-09-22.
# ---------------------------------------------------------------------------

def _bt_param():
    return SimpleNamespace(fb=106.25e9, fb_BT_cutoff=0.75, BTorder=4)


@pytest.mark.parametrize('use_BT', [0, [], [1, 0]])
def test_oracle_matlab_if_semantics_take_the_ones_branch(use_BT):
    """COM Octave: use_BT of [] or [1 0] is FALSE, so the ones branch runs.

    `not use_BT` raised on any numpy array of more than one element, and took
    the FILTER branch for the list [1, 0].
    """
    f = np.array([1e9, 2e9, 3e9])
    np.testing.assert_allclose(
        Bessel_Thomson_Filter(_bt_param(), f, use_BT), np.ones(3))


def test_oracle_all_nonzero_takes_the_filter_branch():
    """COM Octave: use_BT=[1 1] filters; H[0] = 0.99991001458943418-0.012548549092434663j"""
    f = np.array([1e9, 2e9, 3e9])
    H = Bessel_Thomson_Filter(_bt_param(), f, [1, 1])
    np.testing.assert_allclose(
        H[0], 0.99991001458943418 - 0.012548549092434663j, rtol=1e-13)


def test_oracle_length_is_the_longest_dimension():
    """COM Octave: a 2x3 f gives ones(1,3) -- THREE elements, not six.

    MATLAB length() is the longest dimension; len(f) is the first, so a matrix
    axis returned an array of the wrong size.
    """
    fm = np.ones((2, 3)) * np.array([1e9, 2e9, 3e9])
    assert Bessel_Thomson_Filter(_bt_param(), fm, 0).shape == (3,)


def test_oracle_scalar_f():
    """COM Octave: a scalar f with use_BT=0 gives 1; len(f) raised TypeError."""
    np.testing.assert_allclose(
        Bessel_Thomson_Filter(_bt_param(), 1e9, 0), np.ones(1))
