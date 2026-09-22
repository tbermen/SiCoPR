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


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.
#
# pdf.Min = -round(2*nsigma*sigma/binsize), and MATLAB round() sends a half
# away from zero where np.round sends it to even.  Probed on exact halves:
#   normal_dist(0.25,1,1) -> 2*1*0.25/1 = 0.5 -> Min = -1   (np.round: 0)
#   normal_dist(0.75,1,1) ->               1.5 -> Min = -2   (agrees)
#   normal_dist(1.25,1,1) ->               2.5 -> Min = -3   (np.round: -2)
#   normal_dist(1.75,1,1) ->               3.5 -> Min = -4   (agrees)
# The y vectors below are the Octave output for those four calls.
#
# Degenerate arguments:
#   normal_dist(-1.0,2,0.5) -> Min = +8, and (8:-8) is empty, so pdf.x and
#     pdf.y both come back 1x0.  Same for a negative binsize.
#   normal_dist(0.1,1,1), normal_dist(0.0,5,0.1), normal_dist(1e-20,5,0.1)
#     all round to Min = 0 (Octave prints -0) and give x=[0], y=[1].
#
# MEASURED DIVERGENCE, not fixed here: normal_dist(0.01,5,0.001).
#   pdf.x and the pre-normalisation exp(...) vector are bit-identical between
#   Octave and numpy, but sum() is not: Octave accumulates left to right
#   (25.066282746323939, same as a plain Python loop) while np.sum uses
#   pairwise summation (25.066282746323921).  After the divide that leaves
#   pdf.y differing by ~7.6e-16 relative, e.g. y[100] = 0.039894228040121091
#   (Octave) vs 0.039894228040121119 (numpy).  Matching Octave's accumulation
#   would not necessarily match MATLAB's, which is SIMD-blocked and is the
#   real reference, so the ordering is left alone and pinned loosely below.
#   The 17-point case (1.0, 2, 0.5) is short enough that the two agree
#   exactly, and is pinned exactly.
# ============================================================


def test_round_half_away_from_zero_in_min():
    """Exact halves in 2*nsigma*sigma/binsize: MATLAB rounds away from zero."""
    assert normal_dist(0.25, 1, 1.0).Min == -1     # np.round(0.5) would give 0
    assert normal_dist(0.75, 1, 1.0).Min == -2
    assert normal_dist(1.25, 1, 1.0).Min == -3     # np.round(2.5) would give -2
    assert normal_dist(1.75, 1, 1.0).Min == -4


def test_half_tie_pdf_values():
    """Octave pdf.y for the two ties np.round would take to even."""
    np.testing.assert_allclose(
        normal_dist(0.25, 1, 1.0).y,
        [0.00033523770845721439, 0.99932952458308533, 0.00033523770845721439],
        rtol=1e-15)
    np.testing.assert_allclose(
        normal_dist(1.25, 1, 1.0).y,
        [0.017988208587689281, 0.089096180391607646, 0.23269218012423165,
         0.32044686179294296, 0.23269218012423165, 0.089096180391607646,
         0.017988208587689281],
        rtol=1e-15)


def test_exact_match_on_short_axis():
    """17 points: Octave's sum and np.sum agree, so pdf.y matches bit for bit."""
    p = normal_dist(1.0, 2, 0.5)
    assert p.Min == -8
    assert list(p.y[:4]) == [6.6916289572635531e-05, 0.00043634902050678832,
                             0.0022159631725965552, 0.0087643043627858696]
    assert p.y[8] == 0.199474647864745


def test_negative_sigma_gives_empty_axis():
    """Min becomes positive, (Min:-Min) is empty, so x and y are both empty."""
    p = normal_dist(-1.0, 2, 0.5)
    assert p.Min == 8
    assert len(np.atleast_1d(p.x)) == 0
    assert len(np.atleast_1d(p.y)) == 0


def test_negative_binsize_gives_empty_axis():
    p = normal_dist(1.0, 2, -0.5)
    assert p.Min == 8
    assert len(np.atleast_1d(p.y)) == 0


def test_rounds_to_single_bin():
    """2*nsigma*sigma/binsize rounding to 0 leaves a one-point delta."""
    for args in [(0.1, 1, 1.0), (0.0, 5, 0.1), (1.0, 0, 0.1), (1e-20, 5, 0.1)]:
        p = normal_dist(*args)
        assert p.Min == 0
        np.testing.assert_allclose(p.x, [0.0])
        np.testing.assert_allclose(p.y, [1.0])


def test_x_axis_is_bit_identical_to_octave():
    """pdf.x = (Min:-Min)*binsize reproduces exactly; only sum() diverges."""
    p = normal_dist(0.01, 5, 0.001)
    assert p.x[0] == -0.10000000000000001
    assert p.x[43] == -0.057000000000000002
    assert p.x[200] == 0.10000000000000001
