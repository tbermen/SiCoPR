"""Verification tests for conv_fct_MeanNotZero().

# ============================================================
# MATLAB GROUND TRUTH
# Identical logic to conv_fct: p.Min=round(p1.Min+p2.Min),
# p.y=conv(p1.y,p2.y), p.x=(p.Min*BinSize:BinSize:pMax*BinSize).
# Name indicates use for non-zero-mean PDFs (non-symmetric Min).
#
# p1.Min=2, p1.y=[0.6,0.4]; p2.Min=1, p2.y=[0.7,0.3]; BinSize=0.1
#   p.Min = 3
#   p.y = conv([0.6,0.4],[0.7,0.3]) = [0.42,0.46,0.12]
#   pMax = 3+3-1 = 5
#   p.x = (3:5)*0.1 = [0.3, 0.4, 0.5]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.conv_fct_MeanNotZero.py_impl import conv_fct_MeanNotZero


def make_pdf(min_idx, y, binsize=0.1):
    return SimpleNamespace(Min=min_idx, BinSize=binsize, y=np.array(y, dtype=float))


def test_positive_min_convolve():
    """p1.Min=2, p2.Min=1: p.Min=3, y=conv([0.6,0.4],[0.7,0.3])=[0.42,0.46,0.12]."""
    out = conv_fct_MeanNotZero(make_pdf(2, [0.6, 0.4]), make_pdf(1, [0.7, 0.3]))
    assert out.Min == 3
    np.testing.assert_allclose(out.y, [0.42, 0.46, 0.12], atol=1e-12)
    np.testing.assert_allclose(out.x, [0.3, 0.4, 0.5], atol=1e-12)


def test_x_length_matches_y():
    out = conv_fct_MeanNotZero(make_pdf(1, [0.5, 0.5]), make_pdf(2, [0.4, 0.6]))
    assert len(out.x) == len(out.y)


def test_min_additive():
    out = conv_fct_MeanNotZero(make_pdf(-3, [1.0]), make_pdf(5, [1.0]))
    assert out.Min == 2


def test_output_length():
    out = conv_fct_MeanNotZero(make_pdf(0, [0.3, 0.4, 0.3]), make_pdf(0, [0.5, 0.5]))
    assert len(out.y) == 4


def test_binsize_mismatch_raises():
    p1 = make_pdf(1, [1.0], binsize=0.1)
    p2 = make_pdf(1, [1.0], binsize=0.2)
    with pytest.raises(ValueError):
        conv_fct_MeanNotZero(p1, p2)


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py). Same three as conv_fct, since the bodies match:
#
# 1. p.x is a FLOATING-POINT COLON. The MATLAB comment immediately above the
#    line claims it is "equivalent to (p.Min:p.Min+length(p.y)-1)*p.BinSize";
#    executing it shows otherwise. Over 7920 (Min, length, BinSize) cases the
#    product form got 249387 of 1013684 elements wrong, each by 1 ulp.
# 2. p.Min used Python round() (half-to-even) where MATLAB rounds half away
#    from zero.
# 3. An empty operand: conv2 returns empty, np.convolve raised ValueError.
# ============================================================

def test_x_axis_is_a_colon_not_a_product():
    """COM Octave: BinSize=1e-4, p.Min=-5, 9 bins.

    Exact equality on purpose -- the old form was within 1 ulp everywhere, so
    assert_allclose passes on the defect.
    """
    out = conv_fct_MeanNotZero(make_pdf(-2, [1, 2, 3, 2, 1], binsize=1e-4),
                               make_pdf(-3, [1, 2, 3, 2, 1], binsize=1e-4))
    assert out.Min == -5
    assert list(out.x) == [-0.00050000000000000001, -0.00040000000000000002,
                           -0.00030000000000000003, -0.00019999999999999998,
                           -9.9999999999999991e-05, 0,
                           0.00010000000000000005, 0.00019999999999999998,
                           0.00030000000000000003]
    assert list(out.x) != list(np.arange(-5, 4) * 1e-4)


def test_x_last_element_pinned_only_when_it_overshoots():
    """COM Octave: BinSize=0.1, p.Min=-3, 2 bins -> ends -0.20000000000000004."""
    out = conv_fct_MeanNotZero(make_pdf(-3, [1, 1]), make_pdf(0, [1]))
    assert list(out.x) == [-0.30000000000000004, -0.20000000000000004]
    assert out.x[-1] != -2 * 0.1


def test_min_round_is_half_away_from_zero():
    """COM Octave: round(-0.5) = -1; Python round() gave 0."""
    assert conv_fct_MeanNotZero(make_pdf(-0.5, [1, 1]), make_pdf(0, [1])).Min == -1
    assert conv_fct_MeanNotZero(make_pdf(0.5, [1, 1]), make_pdf(0, [1])).Min == 1


def test_empty_operand_returns_empty():
    """COM Octave: conv2([1 2 3], []) is 0x0; np.convolve raised instead."""
    out = conv_fct_MeanNotZero(make_pdf(-1, [1, 2, 3]), make_pdf(0, []))
    assert out.Min == -1
    assert len(out.y) == 0
    assert len(out.x) == 0
