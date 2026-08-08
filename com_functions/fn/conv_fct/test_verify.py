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
