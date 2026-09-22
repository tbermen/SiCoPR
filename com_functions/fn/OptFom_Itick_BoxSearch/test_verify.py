"""Verification tests for OptFom_Itick_BoxSearch().

# ============================================================
# MATLAB GROUND TRUTH (lines 3580-3605)
# itickn <= CL: itick = cluster[itickn-1] (1-based)
# itickn = CL+1: compute BEST.cluster from itick_in_cluster
# itickn > CL+1, BEST.cluster empty → skip_it=1, itick=[]
# itickn > CL+1, BEST.cluster not empty → itick = BEST.cluster[itickn-CL-1]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Itick_BoxSearch.py_impl import OptFom_Itick_BoxSearch


def _best(itick_in_cluster=5, cluster=None):
    return SimpleNamespace(itick_in_cluster=itick_in_cluster,
                           cluster=np.array([]) if cluster is None else np.array(cluster))


def test_first_iteration_returns_cluster_midpoint():
    """itickn=1 → returns cluster[0]."""
    cluster = np.array([3, 8, 13])
    BEST = _best()
    itick, _, skip_it = OptFom_Itick_BoxSearch(1, cluster, BEST, 2, 5)
    assert itick == 3
    assert skip_it == 0


def test_mid_iteration_returns_cluster_value():
    """itickn=2 → returns cluster[1]."""
    cluster = np.array([3, 8, 13])
    BEST = _best()
    itick, _, skip_it = OptFom_Itick_BoxSearch(2, cluster, BEST, 2, 5)
    assert itick == 8


def test_cL_plus_1_builds_best_cluster():
    """itickn = CL+1 → BEST.cluster built from itick_in_cluster box."""
    cluster = np.array([3, 8, 13])  # CL=3
    BEST = _best(itick_in_cluster=8)
    itick, BEST_out, _ = OptFom_Itick_BoxSearch(4, cluster, BEST, 2, 5)
    # box_begin=8-2=6, box_end=6+5-1=10
    # setdiff([6,7,8,9,10], [8]) = [6,7,9,10]
    assert 8 not in BEST_out.cluster
    assert 6 in BEST_out.cluster


def test_empty_best_cluster_returns_skip():
    """BEST.cluster empty → skip_it=1."""
    cluster = np.array([3, 8, 13])  # CL=3
    BEST = _best(itick_in_cluster=8, cluster=[])
    _, _, skip_it = OptFom_Itick_BoxSearch(5, cluster, BEST, 2, 5)
    assert skip_it == 1


def test_after_cL_1_returns_from_best_cluster():
    """itickn > CL+1: returns BEST.cluster[pos]."""
    cluster = np.array([5, 10])  # CL=2
    BEST = _best(itick_in_cluster=10, cluster=[6, 7, 9, 11])
    # itickn=4 → CL+1=3, so pos = 4-2-1 = 1 → BEST.cluster[1] = 7
    itick, _, skip_it = OptFom_Itick_BoxSearch(4, cluster, BEST, 2, 5)
    assert itick == 7
    assert skip_it == 0


# ============================================================
# COM Octave oracle values (2026-09-22)
# OptFom_Itick_BoxSearch run verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m via tools/octave_oracle.py
# (the compat body is byte-identical to matlab/com_ieee8023_4p16p0.m).
# There is no sort in this function, so the stable-sort question does not
# arise; what does is 1-based subscripting and MATLAB's propagation of [].
# ============================================================


@pytest.mark.parametrize('itickn', [0, -1])
def test_octave_itickn_below_one_is_a_subscript_error(itickn):
    """COM Octave: cluster(0) and cluster(-1) are errors --
        error: cluster(0): subscripts must be either integers
               1 to (2^63)-1 or logicals
    -- not a wrap to the end of the vector, which is what a bare Python
    cluster[itickn-1] would give (13 and 8 here)."""
    with pytest.raises(IndexError):
        OptFom_Itick_BoxSearch(itickn, np.array([3, 8, 13]), _best(), 2, 5)


def test_octave_empty_itick_in_cluster_skips():
    """COM Octave: BEST.itick_in_cluster=[] (nothing beat the starting FOM,
    the 'every case was bad' path) with itickn=CL+1 gives
    itick=[], skip_it=1, BEST.cluster=[]. MATLAB propagates the empty
    through [] - box_mid and the colon range; it does not error."""
    BEST = _best(itick_in_cluster=[])
    itick, BEST_out, skip_it = OptFom_Itick_BoxSearch(
        4, np.array([3, 8, 13]), BEST, 2, 5)
    assert skip_it == 1
    assert len(itick) == 0
    assert np.size(BEST_out.cluster) == 0


def test_octave_non_integer_midpoint_is_not_truncated():
    """COM Octave: itick_in_cluster=8.5, box_mid=2, box_size=5 ->
    BEST.cluster = [6.5 7.5 9.5 10.5], itick = 6.5."""
    BEST = _best(itick_in_cluster=8.5)
    itick, BEST_out, skip_it = OptFom_Itick_BoxSearch(
        4, np.array([3, 8, 13]), BEST, 2, 5)
    np.testing.assert_allclose(np.asarray(BEST_out.cluster).ravel(),
                               [6.5, 7.5, 9.5, 10.5])
    assert itick == 6.5
    assert skip_it == 0


def test_octave_non_integer_cluster_value_is_not_truncated():
    """COM Octave: cluster=[3.5 8.5], itickn=1 -> itick = 3.5."""
    itick, _, _ = OptFom_Itick_BoxSearch(1, np.array([3.5, 8.5]), _best(), 2, 5)
    assert itick == 3.5


def test_octave_cl_plus_1_box_values():
    """COM Octave: cluster=[3 8 13], itick_in_cluster=8, box_mid=2,
    box_size=5 -> BEST.cluster = [6 7 9 10], itick = 6."""
    BEST = _best(itick_in_cluster=8, cluster=[99, 98])
    itick, BEST_out, skip_it = OptFom_Itick_BoxSearch(
        4, np.array([3, 8, 13]), BEST, 2, 5)
    np.testing.assert_array_equal(np.asarray(BEST_out.cluster).ravel(),
                                  [6, 7, 9, 10])
    assert itick == 6
    assert skip_it == 0


def test_octave_negative_box_mid():
    """COM Octave: box_mid=-2 -> BEST.cluster = [10 11 12 13 14], itick = 10."""
    BEST = _best(itick_in_cluster=8)
    itick, BEST_out, _ = OptFom_Itick_BoxSearch(
        4, np.array([3, 8, 13]), BEST, -2, 5)
    np.testing.assert_array_equal(np.asarray(BEST_out.cluster).ravel(),
                                  [10, 11, 12, 13, 14])
    assert itick == 10


@pytest.mark.parametrize('box_mid,box_size', [(2, 0), (0, 1)])
def test_octave_degenerate_box_skips(box_mid, box_size):
    """COM Octave: an empty box (box_size=0) and a one-element box whose only
    member is the midpoint both give itick=[], skip_it=1."""
    BEST = _best(itick_in_cluster=8)
    itick, _, skip_it = OptFom_Itick_BoxSearch(
        4, np.array([3, 8, 13]), BEST, box_mid, box_size)
    assert skip_it == 1
    assert len(itick) == 0


def test_octave_empty_cluster_goes_to_box_branch():
    """COM Octave: cluster=[] is CL=0, so itickn=1 is CL+1 and the box is
    built at once: itick = 6."""
    BEST = _best(itick_in_cluster=8)
    itick, _, skip_it = OptFom_Itick_BoxSearch(1, np.array([]), BEST, 2, 5)
    assert itick == 6
    assert skip_it == 0


def test_octave_past_end_of_best_cluster_errors():
    """COM Octave: itickn=8 with a 4-element BEST.cluster --
        error: BEST(5): out of bound 4 (dimensions are 1x4)"""
    BEST = _best(itick_in_cluster=8, cluster=[6, 7, 9, 10])
    with pytest.raises(IndexError):
        OptFom_Itick_BoxSearch(8, np.array([3, 8, 13]), BEST, 2, 5)
