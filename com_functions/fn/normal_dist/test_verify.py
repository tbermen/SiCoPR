"""Verification tests for normal_dist() — Gaussian PDF struct.

# ============================================================
# MATLAB GROUND TRUTH
# Derived analytically from the formula:
#
# normal_dist(sigma=0.01, nsigma=5, binsize=0.001):
#   pdf.Min = -round(2*5*0.01/0.001) = -round(100) = -100
#   pdf.x   = (-100:100) * 0.001  →  201 points, x[0]=-0.1, x[-1]=0.1
#   pdf.y   = exp(-x^2 / (2*0.01^2 + eps)), then normalised → sum=1
#
# normal_dist(sigma=1.0, nsigma=2, binsize=0.5):
#   pdf.Min = -round(2*2*1.0/0.5) = -round(8) = -8
#   pdf.x   = (-8:8) * 0.5  →  17 points, x[0]=-4.0, x[-1]=4.0
#   pdf.y normalised, peak at centre index 8
#
# normal_dist(sigma=1.0, nsigma=3, binsize=1.0):
#   pdf.Min = -round(6) = -6
#   pdf.x   = (-6:6)*1.0  →  13 points, x[0]=-6, x[-1]=6
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.normal_dist.py_impl import normal_dist


def test_binsize_stored():
    p = normal_dist(0.01, 5, 0.001)
    assert p.BinSize == pytest.approx(0.001)


def test_min_field():
    """pdf.Min = -round(2*nsigma*sigma/binsize)."""
    p = normal_dist(0.01, 5, 0.001)
    assert p.Min == -100

    p2 = normal_dist(1.0, 2, 0.5)
    assert p2.Min == -8


def test_x_axis_length():
    """pdf.x has 2*|pdf.Min|+1 elements (symmetric, includes zero)."""
    p = normal_dist(0.01, 5, 0.001)
    assert len(p.x) == 201

    p2 = normal_dist(1.0, 2, 0.5)
    assert len(p2.x) == 17


def test_x_axis_endpoints():
    """First and last x values are Min*binsize and -Min*binsize."""
    p = normal_dist(0.01, 5, 0.001)
    assert p.x[0] == pytest.approx(-0.1)
    assert p.x[-1] == pytest.approx(0.1)

    p2 = normal_dist(1.0, 2, 0.5)
    assert p2.x[0] == pytest.approx(-4.0)
    assert p2.x[-1] == pytest.approx(4.0)


def test_normalised():
    """pdf.y must sum to 1.0."""
    for sigma, nsigma, binsize in [(0.01, 5, 0.001), (1.0, 2, 0.5), (0.5, 4, 0.1)]:
        p = normal_dist(sigma, nsigma, binsize)
        assert np.sum(p.y) == pytest.approx(1.0, rel=1e-12)


def test_symmetric():
    """pdf.y is an even function: y[i] == y[-i-1]."""
    p = normal_dist(1.0, 3, 1.0)
    np.testing.assert_allclose(p.y, p.y[::-1], rtol=1e-12)


def test_peak_at_centre():
    """Maximum of pdf.y is at the centre bin (x=0)."""
    p = normal_dist(1.0, 2, 0.5)
    centre = len(p.y) // 2
    assert np.argmax(p.y) == centre


def test_all_values_nonnegative():
    """Gaussian PDF values are all >= 0."""
    p = normal_dist(0.01, 5, 0.001)
    assert np.all(p.y >= 0)
