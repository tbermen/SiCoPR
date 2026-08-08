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
