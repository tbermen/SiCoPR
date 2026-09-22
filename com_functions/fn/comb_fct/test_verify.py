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


# ============================================================
# Probed against the executed reference (COM Octave oracle,
# tools/octave_oracle.py). NO divergence was found in comb_fct's values --
# these pin the behaviour so it stays that way.
#
# Worth contrasting with conv_fct: comb_fct's axis is (p.Min:-p.Min)*BinSize,
# an INTEGER colon scaled afterwards, so np.arange(...)*BinSize is right here.
# conv_fct's axis is a FLOATING-POINT colon and that same form is wrong there.
# ============================================================

def test_octave_shift_branches():
    """COM Octave, BinSize=0.1: whichever side has the smaller Min, the other
    is zero-padded into its frame and the sum is [1 3 5 3 1]."""
    p1 = SimpleNamespace(Min=-2, BinSize=0.1, y=np.array([1., 2, 3, 2, 1]))
    p2 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1., 2, 1]))
    out = comb_fct(p1, p2)
    assert out.Min == -2
    np.testing.assert_array_equal(out.y, [1, 3, 5, 3, 1])
    out = comb_fct(p2, p1)          # same pair, swapped
    assert out.Min == -2
    np.testing.assert_array_equal(out.y, [1, 3, 5, 3, 1])


def test_octave_equal_min_shorter_second():
    """COM Octave: equal Min with a shorter p2 zero-fills p2 out to len(p1)
    -> [2 3 3 2 1]."""
    p1 = SimpleNamespace(Min=-2, BinSize=0.1, y=np.array([1., 2, 3, 2, 1]))
    p2 = SimpleNamespace(Min=-2, BinSize=0.1, y=np.array([1., 1]))
    np.testing.assert_array_equal(comb_fct(p1, p2).y, [2, 3, 3, 2, 1])


def test_octave_x_axis_is_an_integer_colon_scaled():
    """COM Octave, BinSize=1e-4, p.Min=-4 -> x is exactly
    np.arange(-4,5)*1e-4, including -0.0001 (not the -9.9999...e-05 that
    conv_fct's floating-point colon produces at the same bin)."""
    p1 = SimpleNamespace(Min=-4, BinSize=1e-4,
                         y=np.array([1., 2, 3, 2, 1, 1, 1, 1, 1]))
    p2 = SimpleNamespace(Min=-2, BinSize=1e-4, y=np.array([1., 2, 1]))
    out = comb_fct(p1, p2)
    assert list(out.x) == [-0.00040000000000000002, -0.00030000000000000003,
                           -0.00020000000000000001, -0.0001, 0, 0.0001,
                           0.00020000000000000001, 0.00030000000000000003,
                           0.00040000000000000002]
    np.testing.assert_array_equal(out.x, np.arange(-4, 5) * 1e-4)


def test_octave_positive_min_gives_an_empty_axis():
    """COM Octave: p.Min=2 makes (2:-2) empty, so p.x is 1x0 while p.y is not.
    p.Min=0 gives the single point 0."""
    p1 = SimpleNamespace(Min=2, BinSize=0.1, y=np.array([1., 2, 3]))
    p2 = SimpleNamespace(Min=2, BinSize=0.1, y=np.array([1., 1, 1]))
    out = comb_fct(p1, p2)
    np.testing.assert_array_equal(out.y, [2, 3, 4])
    assert len(out.x) == 0
    p1.Min = p2.Min = 0
    np.testing.assert_array_equal(comb_fct(p1, p2).x, [0.0])


def test_octave_nonconformant_lengths_raise():
    """COM Octave: when the padded operands end up different lengths the sum
    is 'operator +: nonconformant arguments'. Python must refuse too."""
    p1 = SimpleNamespace(Min=-2, BinSize=0.1, y=np.array([1., 2]))
    p2 = SimpleNamespace(Min=-2, BinSize=0.1, y=np.array([1., 1, 1, 1, 1]))
    with pytest.raises(ValueError):
        comb_fct(p1, p2)
