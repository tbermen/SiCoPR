"""Verification tests for comb_fct().

# ============================================================
# MATLAB GROUND TRUTH
# PDF struct: Min (negative int bin index), BinSize, y (probability array)
# p.x = (p.Min:-p.Min)*BinSize  (symmetric axis of length 2*|Min|+1)
#
# Case: p1.Min=p2.Min=-1, y1=[0.5,1,0.5], y2=[0.3,0.6,0.3], BinSize=0.1
#   Same Min → neither branch; p.y = y1+y2 = [0.8,1.6,0.8]
#   p.x = (-1:1)*0.1 = [-0.1,0,0.1]
#
# Case: p1.Min=-2, y1=[1,2,3,2,1]; p2.Min=-1, y2=[1,2,1]; BinSize=0.1
#   p1.Min == p.Min (-2): expand p2 right by difsz=1
#   p2_new = [0,1,2,1,0]  (length 5 = lp1)
#   p.y = [1,2,3,2,1]+[0,1,2,1,0] = [1,3,5,3,1]
#   p.x = (-2:2)*0.1 = [-0.2,-0.1,0,0.1,0.2]
#
# Case: p1.Min=-1, y1=[1,2,1]; p2.Min=-2, y2=[1,2,3,2,1]; BinSize=0.1
#   p2.Min == p.Min (-2): expand p1 left by difsz=1
#   p1_new = [0,1,2,1,0]  (length 5 = lp2)
#   p.y = [0,1,2,1,0]+[1,2,3,2,1] = [1,3,5,3,1]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.comb_fct.py_impl import comb_fct


def make_pdf(min_idx, y, binsize=0.1):
    return SimpleNamespace(Min=min_idx, BinSize=binsize,
                           y=np.array(y, dtype=float))


def test_same_min_direct_addition():
    """Equal Min → y = y1 + y2, no shifting."""
    p1 = make_pdf(-1, [0.5, 1.0, 0.5])
    p2 = make_pdf(-1, [0.3, 0.6, 0.3])
    out = comb_fct(p1, p2)
    np.testing.assert_allclose(out.y, [0.8, 1.6, 0.8])
    np.testing.assert_allclose(out.x, [-0.1, 0.0, 0.1])


def test_p1_has_smaller_min():
    """p1.Min < p2.Min: p2 shifted right, zero-padded to match p1 length."""
    p1 = make_pdf(-2, [1, 2, 3, 2, 1])
    p2 = make_pdf(-1, [1, 2, 1])
    out = comb_fct(p1, p2)
    np.testing.assert_allclose(out.y, [1, 3, 5, 3, 1])
    assert out.Min == -2
    np.testing.assert_allclose(out.x, [-0.2, -0.1, 0.0, 0.1, 0.2])


def test_p2_has_smaller_min():
    """p2.Min < p1.Min: p1 shifted right, zero-padded to match p2 length."""
    p1 = make_pdf(-1, [1, 2, 1])
    p2 = make_pdf(-2, [1, 2, 3, 2, 1])
    out = comb_fct(p1, p2)
    np.testing.assert_allclose(out.y, [1, 3, 5, 3, 1])
    assert out.Min == -2


def test_binsize_mismatch_raises():
    """BinSize mismatch → ValueError."""
    p1 = make_pdf(-1, [1, 2, 1], binsize=0.1)
    p2 = make_pdf(-1, [1, 2, 1], binsize=0.2)
    with pytest.raises(ValueError):
        comb_fct(p1, p2)


def test_x_axis_correct_length():
    """len(p.x) == 2*|p.Min|+1."""
    p1 = make_pdf(-3, np.ones(7))
    p2 = make_pdf(-3, np.ones(7))
    out = comb_fct(p1, p2)
    assert len(out.x) == 2 * abs(out.Min) + 1
