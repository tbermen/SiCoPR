"""Verification tests for conv_fct().

# ============================================================
# MATLAB GROUND TRUTH
# p.Min = round(p1.Min + p2.Min)
# p.y = conv(p1.y, p2.y)   (N1+N2-1 elements)
# pMax = p.Min + length(p.y) - 1
# p.x = (p.Min*BinSize : BinSize : pMax*BinSize)
#
# p1.Min=-1, p1.y=[1,1]; p2.Min=-1, p2.y=[1,1]; BinSize=0.1
#   p.Min = -2
#   p.y = conv([1,1],[1,1]) = [1,2,1]
#   pMax = -2 + 3 - 1 = 0
#   p.x = (-2:0)*0.1 = [-0.2,-0.1,0]
#
# p1.Min=0, p1.y=[1]; p2.Min=0, p2.y=[1]; BinSize=0.1
#   p.Min=0, p.y=[1], p.x=[0]
#
# Length: len(p.y) == len(p.x) always
# BinSize preserved from p1
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.conv_fct.py_impl import conv_fct


def make_pdf(min_idx, y, binsize=0.1):
    return SimpleNamespace(Min=min_idx, BinSize=binsize, y=np.array(y, dtype=float))


def test_symmetric_convolve():
    """[1,1]*[1,1]=[1,2,1]; Min=-1+-1=-2; x=[-0.2,-0.1,0]."""
    out = conv_fct(make_pdf(-1, [1, 1]), make_pdf(-1, [1, 1]))
    np.testing.assert_allclose(out.y, [1, 2, 1])
    assert out.Min == -2
    np.testing.assert_allclose(out.x, [-0.2, -0.1, 0.0])


def test_single_element():
    """[1]*[1]=[1]; Min=0; x=[0]."""
    out = conv_fct(make_pdf(0, [1.0]), make_pdf(0, [1.0]))
    np.testing.assert_allclose(out.y, [1.0])
    assert out.Min == 0
    np.testing.assert_allclose(out.x, [0.0])


def test_min_additive():
    """p.Min = p1.Min + p2.Min."""
    out = conv_fct(make_pdf(-3, [0.5, 0.5]), make_pdf(-2, [0.5, 0.5]))
    assert out.Min == -5


def test_output_length():
    """len(p.y) = len(p1.y) + len(p2.y) - 1."""
    out = conv_fct(make_pdf(-2, [0.1, 0.5, 0.4]), make_pdf(-1, [0.3, 0.7]))
    assert len(out.y) == 4


def test_x_length_matches_y():
    """len(p.x) == len(p.y) always."""
    out = conv_fct(make_pdf(-2, [0.2, 0.6, 0.2]), make_pdf(-2, [0.3, 0.4, 0.3]))
    assert len(out.x) == len(out.y)


def test_binsize_mismatch_raises():
    p1 = make_pdf(-1, [1, 1], binsize=0.1)
    p2 = make_pdf(-1, [1, 1], binsize=0.2)
    with pytest.raises(ValueError):
        conv_fct(p1, p2)


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py against octave/com_ieee8023_4p16p0_octave_compat.m).
# Values below are what Octave returned, pinned as literals.
#
# 1. p.x is a FLOATING-POINT COLON, not an integer range times BinSize.
#    The MATLAB comment in conv_fct_MeanNotZero calls the two "equivalent";
#    they are not. Swept over 7920 (Min, length, BinSize) combinations, the
#    product form np.arange(Min, pMax+1)*BinSize got 249387 of 1013684
#    elements wrong -- 24.6%, each by 1 ulp.
# 2. p.Min used Python round(), which is half-to-even; MATLAB round() is
#    half-away-from-zero, so Min=0.5 gave 0 instead of 1.
# 3. An empty operand: conv2 returns empty, np.convolve raised ValueError.
# ============================================================

def test_x_axis_is_a_colon_not_a_product():
    """COM Octave: BinSize=1e-5, p.Min=-7, 4 bins.

    Exact equality on purpose: every element of the old product form was
    within 1 ulp, so assert_allclose passes on the defect.
    """
    out = conv_fct(make_pdf(-3, [1, 1, 1], binsize=1e-5),
                   make_pdf(-4, [1, 1], binsize=1e-5))
    assert out.Min == -7
    np.testing.assert_array_equal(out.y, [1, 2, 2, 1])
    assert list(out.x) == [-7.0000000000000007e-05, -6.0000000000000008e-05,
                           -5.0000000000000009e-05, -4.0000000000000003e-05]
    # the middle bin is where the product form lands 1 ulp away
    assert out.x[2] != np.arange(-7, -3)[2] * 1e-5


def test_x_axis_colon_nine_bins():
    """COM Octave: BinSize=1e-4, p.Min=-5, 9 bins."""
    out = conv_fct(make_pdf(-2, [1, 2, 3, 2, 1], binsize=1e-4),
                   make_pdf(-3, [1, 2, 3, 2, 1], binsize=1e-4))
    assert out.Min == -5
    assert list(out.x) == [-0.00050000000000000001, -0.00040000000000000002,
                           -0.00030000000000000003, -0.00019999999999999998,
                           -9.9999999999999991e-05, 0,
                           0.00010000000000000005, 0.00019999999999999998,
                           0.00030000000000000003]


def test_x_last_element_pinned_only_when_it_overshoots():
    """COM Octave: BinSize=0.1, p.Min=-3, 2 bins -> ends -0.20000000000000004.

    Accumulation undershoots the limit here, so the colon keeps the
    accumulated value; forcing the last element to pMax*BinSize gives -0.2.
    """
    out = conv_fct(make_pdf(-3, [1, 1]), make_pdf(0, [1]))
    assert out.Min == -3
    assert list(out.x) == [-0.30000000000000004, -0.20000000000000004]
    assert out.x[-1] != -2 * 0.1


def test_x_axis_kernel_longer_than_signal():
    """COM Octave: p1.y 2 bins, p2.y 5 bins, BinSize=0.1, p.Min=-4."""
    out = conv_fct(make_pdf(-1, [1, 2]), make_pdf(-3, [1, 2, 3, 4, 5]))
    assert out.Min == -4
    np.testing.assert_array_equal(out.y, [1, 4, 7, 10, 13, 10])
    assert list(out.x) == [-0.40000000000000002, -0.30000000000000004,
                           -0.20000000000000001, -0.099999999999999978,
                           0, 0.099999999999999978]


def test_min_round_is_half_away_from_zero():
    """COM Octave: round(0.5)=1, round(-0.5)=-1, round(2.5)=3.

    Python's round() is half-to-even and gave 0, 0 and 2.
    """
    assert conv_fct(make_pdf(0.5, [1, 1]), make_pdf(0, [1])).Min == 1
    assert conv_fct(make_pdf(-0.5, [1, 1]), make_pdf(0, [1])).Min == -1
    assert conv_fct(make_pdf(2.5, [1, 1]), make_pdf(0, [1])).Min == 3


def test_empty_operand_returns_empty():
    """COM Octave: conv2([1 2 3], []) is 0x0, so p.y and p.x are empty and
    p.Min is still round(p1.Min+p2.Min). np.convolve raised instead."""
    out = conv_fct(make_pdf(-1, [1, 2, 3]), make_pdf(0, []))
    assert out.Min == -1
    assert len(out.y) == 0
    assert len(out.x) == 0
    out = conv_fct(make_pdf(-1, []), make_pdf(-2, [1, 2, 3]))
    assert out.Min == -3
    assert len(out.y) == 0
    assert len(out.x) == 0
