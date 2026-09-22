"""Verification tests for d_cpdf().

# ============================================================
# MATLAB GROUND TRUTH
# Special case all(values==0):
#   pdf.BinSize=binsize, pdf.Min=0, pdf.y=1, pdf.x=0
#
# Symmetric 3-point: values=[-0.1,0,0.1], probs=[0.25,0.5,0.25], binsize=0.1
#   t = [-0.1, 0, 0.1]  (after snap, same)
#   pdf.Min = -1
#   k=0: bin=0 → pdf.y[0]+=0.25
#   k=1: argmin(|t-0|)=1 → pdf.y[1]+=0.5
#   k=2: bin=last=2 → pdf.y[2]+=0.25
#   pdf.y = [0.25,0.5,0.25] (sum=1, already normalized)
#   support = [0,1,2] → no trim
#   pdf.Min = -1+(0) = -1
#   pdf.x = (-1:1)*0.1 = [-0.1,0,0.1]
#
# Normalization: probs=[1,3] → pdf.y=[0.25,0,0.75] after normalize
#   (middle bin stays 0 since only first and last bin are populated)
#
# Unsorted input: same result as sorted (argsort applied)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.d_cpdf.py_impl import d_cpdf


def test_all_zeros_special_case():
    """all(values==0) → scalar y=1, x=0."""
    out = d_cpdf(0.1, np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.3, 0.2]))
    assert out.Min == 0
    assert out.BinSize == pytest.approx(0.1)
    np.testing.assert_allclose(out.y, [1.0])
    np.testing.assert_allclose(out.x, [0.0])


def test_symmetric_three_point():
    """values=[-0.1,0,0.1]: pdf.y=[0.25,0.5,0.25], pdf.x=[-0.1,0,0.1]."""
    out = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    np.testing.assert_allclose(out.y, [0.25, 0.5, 0.25], atol=1e-12)
    np.testing.assert_allclose(out.x, [-0.1, 0.0, 0.1], atol=1e-12)
    assert out.Min == -1


def test_normalization():
    """Unnormalized probs → pdf.y sums to 1."""
    out = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([1.0, 2.0, 1.0]))
    assert np.sum(out.y) == pytest.approx(1.0, rel=1e-12)


def test_unsorted_same_as_sorted():
    """Unsorted values should give same result as sorted."""
    out_sorted = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    out_unsorted = d_cpdf(0.1, np.array([0.1, -0.1, 0.0]), np.array([0.25, 0.25, 0.5]))
    np.testing.assert_allclose(out_sorted.y, out_unsorted.y, atol=1e-12)


def test_binsize_stored():
    out = d_cpdf(0.05, np.array([-0.1, 0.0, 0.1]), np.array([1.0, 2.0, 1.0]))
    assert out.BinSize == pytest.approx(0.05)


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.
#
# support = find(pdf.y) selects *nonzero*, and NaN is nonzero:
#   d_cpdf(1,[-1 0 1],[0.5 NaN 0.5]) -> Min=-1, x=[-1 0 1], y=[NaN NaN NaN]
#   d_cpdf(1,[-1 0 1],[0 0 0])       -> same (pdf.y/sum(pdf.y) is 0/0)
#   The `pdf_y > 0` form dropped NaN, left the support empty and raised.
#
# probs must be at least as long as values:
#   d_cpdf(1,[-1 0 1],[0.5 0.5]) errors
#     "probs(3): out of bound 2 (dimensions are 1x2)"
#   zip(values, probs) stopped at the shorter one and normalised what it had,
#   answering Min=-1, y=[0.5 0.5].
#   A longer probs is fine: the loop only runs to length(values), so
#   d_cpdf(1,[-1 0 1],[0.25 0.25 0.25 0.25]) -> y=[1/3 1/3 1/3].
#
# pdf.x is (pdf.Min:-pdf.Min)*binsize, which has nothing to do with the
# length of pdf.y.  It is EMPTY whenever pdf.Min > 0:
#   d_cpdf(1,[1 2 3],[0.2 0.5 0.3])   -> Min=1,  x=1x0,      y=[0.2 0.5 0.3]
#   d_cpdf(1,[0.5],[1])               -> Min=1,  x=1x0,      y=[1]
#   d_cpdf(1,[-1 0 1 2 3],[.1 .2 .4 .2 .1])
#                                     -> Min=-1, x=[-1 0 1] (3), y len 5
#
# round() half away from zero, on the values/binsize snap:
#   d_cpdf(1,[-2.5 -1.5 -0.5],[0.2 0.5 0.3]) -> Min=-3, x=[-3..3]
#     (np.round would give -2.0,-2.0,-0.0 and a different Min)
#   d_cpdf(1,[0.5 1.5 2.5],[0.2 0.5 0.3])    -> Min=1,  x=1x0
#
# issorted() with a NaN: MATLAB requires every element <= the next, which a
# NaN never satisfies, so [-1 NaN 1] is NOT sorted and gets sorted to
# [-1 1 NaN].  np.diff(values) < 0 is False across a NaN and called it
# sorted, which let Python answer y=[0.6 0 0.4].  Octave then builds
# t = (-1:1:NaN), which it evaluates as the single element [NaN], and ends up
# returning the degenerate Min=-1, x=[-1 0 1], y=[1].  That colon behaviour
# is not reproduced here — Python declines instead of inventing a number.
# ============================================================


def test_nan_prob_propagates():
    """find(pdf.y) counts NaN as nonzero, so the whole pdf comes back NaN."""
    out = d_cpdf(1.0, np.array([-1.0, 0.0, 1.0]), np.array([0.5, np.nan, 0.5]))
    assert out.Min == -1
    np.testing.assert_allclose(out.x, [-1.0, 0.0, 1.0])
    assert len(out.y) == 3
    assert np.all(np.isnan(out.y))


def test_all_probs_zero_gives_nan():
    """sum(pdf.y) is 0, so pdf.y/sum is NaN everywhere — not an empty support."""
    out = d_cpdf(1.0, np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
    assert out.Min == -1
    assert np.all(np.isnan(out.y))


def test_probs_shorter_than_values_refused():
    with pytest.raises(IndexError):
        d_cpdf(1.0, np.array([-1.0, 0.0, 1.0]), np.array([0.5, 0.5]))
    with pytest.raises(IndexError):
        d_cpdf(1.0, np.array([-2.0, -1.0, 0.0, 1.0]), np.array([0.25, 0.25, 0.25]))


def test_probs_longer_than_values_is_allowed():
    out = d_cpdf(1.0, np.array([-1.0, 0.0, 1.0]),
                 np.array([0.25, 0.25, 0.25, 0.25]))
    np.testing.assert_allclose(out.y, [1 / 3.0, 1 / 3.0, 1 / 3.0], rtol=1e-15)


def test_nan_value_refused():
    """A NaN value makes the reference sort it last and build t = (a:bin:NaN)."""
    with pytest.raises(ValueError):
        d_cpdf(1.0, np.array([-1.0, np.nan, 1.0]), np.array([0.3, 0.3, 0.4]))


def test_x_is_empty_when_min_positive():
    """(Min:-Min) is empty for Min>0, so pdf.x is empty while pdf.y is not."""
    out = d_cpdf(1.0, np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.5, 0.3]))
    assert out.Min == 1
    assert len(np.atleast_1d(out.x)) == 0
    np.testing.assert_allclose(out.y, [0.2, 0.5, 0.3])

    single = d_cpdf(1.0, np.array([0.5]), np.array([1.0]))
    assert single.Min == 1
    assert len(np.atleast_1d(single.x)) == 0
    np.testing.assert_allclose(single.y, [1.0])


def test_x_length_need_not_match_y():
    """pdf.x spans (Min:-Min); pdf.y spans the populated bins."""
    out = d_cpdf(1.0, np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
                 np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
    assert out.Min == -1
    assert len(out.x) == 3
    assert len(out.y) == 5


def test_round_half_away_from_zero_on_values():
    """Exact halves snap away from zero, which moves pdf.Min."""
    neg = d_cpdf(1.0, np.array([-2.5, -1.5, -0.5]), np.array([0.2, 0.5, 0.3]))
    assert neg.Min == -3
    np.testing.assert_allclose(neg.x, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(neg.y, [0.2, 0.5, 0.3], rtol=1e-15)

    pos = d_cpdf(1.0, np.array([0.5, 1.5, 2.5]), np.array([0.2, 0.5, 0.3]))
    assert pos.Min == 1
    np.testing.assert_allclose(pos.y, [0.2, 0.5, 0.3], rtol=1e-15)


def test_leading_zero_bins_are_trimmed():
    """support(1) trims leading zeros and shifts pdf.Min with it."""
    out = d_cpdf(1.0, np.array([-3.0, -2.0, 0.0, 2.0]),
                 np.array([0.0, 0.0, 0.5, 0.5]))
    assert out.Min == 0
    np.testing.assert_allclose(out.x, [0.0])
    np.testing.assert_allclose(out.y, [0.5, 0.0, 0.5])
