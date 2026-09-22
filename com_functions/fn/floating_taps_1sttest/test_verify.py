"""Verification tests for floating_taps_1sttest().

# ============================================================
# MATLAB GROUND TRUTH
# [bmax, floating_tap_locations] = floating_taps_1sttest(hisi,N_b,N_bf,N_bg,N_bmax,bmaxg)
# N_bg=0: bmax=0, floating_tap_locations=[]
# N_bg=1: find contiguous N_bf taps starting at index in [N_b..N_bmax-N_bf] that
#         minimise norm(hrem(hisi,ig,N_bf,bmaxg))
# floating_tap_locations is sorted ascending, 0-based Python indices
# bmax[floating_tap_locations] == bmaxg
# len(bmax) == N_bmax
#
# hrem(h,i,N_bf,bmaxg): subtract min(bmaxg,|h[i:i+N_bf]|)*sign from N_bf taps at i
#   With bmaxg >= max|h|: hrem zeroes the N_bf taps → minimal norm
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.floating_taps_1sttest.py_impl import floating_taps_1sttest


def test_n_bg_zero_leaves_the_second_output_unassigned():
    """N_bg=0 returns on `bmax=0` without ever assigning the locations.

    COM Octave: `[bmax,locs] = floating_taps_1sttest(...,0,...)` ->
    "error: element number 2 undefined in return list"; with one output
    bmax is the scalar 0.  This port always returns both, so it refuses.
    """
    hisi = np.array([0.1, 0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match='N_bg=0'):
        floating_taps_1sttest(hisi, 1, 1, 0, 4, 0.5)


def test_one_group_finds_largest_tap():
    """N_bg=1, tap_bk=1: best group is the single largest in-range tap."""
    hisi = np.array([0.0, 0.1, 0.9, 0.2, 0.0, 0.0])
    # N_b=1 (0-based start), N_bf=1, N_bmax=5 → search range [1,4)
    bmax, locs = floating_taps_1sttest(hisi, 1, 1, 1, 5, 2.0)
    assert len(locs) == 1
    assert locs[0] == 2   # index of 0.9


def test_bmax_set_at_locations():
    """bmax[locs] == bmaxg for all found locations."""
    hisi = np.array([0.0, 0.1, 0.8, 0.3, 0.0, 0.0])
    bmaxg = 0.6
    bmax, locs = floating_taps_1sttest(hisi, 1, 1, 1, 5, bmaxg)
    np.testing.assert_allclose(bmax[locs], bmaxg)


def test_locs_sorted():
    """floating_tap_locations is sorted ascending."""
    hisi = np.random.rand(10)
    bmax, locs = floating_taps_1sttest(hisi, 1, 1, 2, 9, 0.5)
    assert np.all(np.diff(locs) >= 0)


def test_bmax_length():
    """len(bmax) == N_bmax."""
    hisi = np.ones(8)
    bmax, _ = floating_taps_1sttest(hisi, 1, 1, 1, 7, 0.5)
    assert len(bmax) == 7


def test_two_groups_two_locs():
    """N_bg=2, tap_bk=1: find 2 locations."""
    hisi = np.array([0.0, 0.5, 0.1, 0.8, 0.0, 0.0, 0.0])
    _, locs = floating_taps_1sttest(hisi, 1, 1, 2, 6, 1.0)
    assert len(locs) == 2


# ============================================================
# COM Octave oracle values (2026-09-22)
# floating_taps_1sttest and hrem extracted verbatim from the reference and run
# under Octave on a fixed 20-tap hisi.  MATLAB's locations are 1-based; this
# port is 0-based throughout (see KNOWN_BEHAVIOUR), so the pins below are the
# Octave locations minus one.
# ============================================================

OCT_HISI = np.array([0.0001, 0.0299, -0.0274, -0.0891, -0.0455, -0.0992, 0.006,
                     0.134, -0.0492, -0.062, 0.049, 0.0357, 0.0105, -0.093,
                     -0.0029, 0.0695, -0.1344, -0.0458, -0.1901, -0.129])


@pytest.mark.parametrize('N_bg,coop,oct_locs', [
    (1, 0, [6, 7, 8]),
    (2, 0, [3, 4, 5, 6, 7, 8]),
    (3, 0, [3, 4, 5, 6, 7, 8, 12, 13, 14]),
    (1, 1, [6, 7, 8]),
    (2, 1, [4, 5, 6, 8, 9, 10]),
    (3, 1, [4, 5, 6, 8, 9, 10, 12, 13, 14]),
])
def test_oracle_group_placement(N_bg, coop, oct_locs):
    """N_b=2, N_bf=3, N_bmax=16, bmaxg=0.2.  Sequential and co-optimised
    searches pick different banks, and COM Octave pins both."""
    bmax, locs = floating_taps_1sttest(OCT_HISI, 2, 3, N_bg, 16, 0.2, coop)
    want = np.array(oct_locs) - 1
    np.testing.assert_array_equal(locs, want)
    expect = np.zeros(16)
    expect[want] = 0.2
    np.testing.assert_array_equal(bmax, expect)


def test_oracle_column_hisi_is_transposed():
    """iscolumn(hisi) -> hisi.'  COM Octave gives the same answer as the row."""
    bmax, locs = floating_taps_1sttest(OCT_HISI.reshape(-1, 1), 2, 3, 1, 16, 0.2, 0)
    np.testing.assert_array_equal(locs, [5, 6, 7])


def test_oracle_ties_keep_the_first_group():
    """`sigma < best_sigma` is strict, so with bmaxg=0 nothing ever improves
    on the first candidate.

    COM Octave, bmaxg=0, N_bg=1 sequential -> locations [3 4 5] (1-based),
    i.e. the very first ig1 = N_b+1.
    """
    bmax, locs = floating_taps_1sttest(OCT_HISI, 2, 3, 1, 16, 0.0, 0)
    np.testing.assert_array_equal(locs, [2, 3, 4])
    np.testing.assert_array_equal(bmax, np.zeros(16))


def test_oracle_coop_groups_may_coincide():
    """Co-optimised banks are not forced apart.

    COM Octave, bmaxg=0, N_bg=2, COOP=1 -> [3 3 4 4 5 5] (1-based): both
    groups land on the same place and the locations repeat.
    """
    _, locs = floating_taps_1sttest(OCT_HISI, 2, 3, 2, 16, 0.0, 1)
    np.testing.assert_array_equal(locs, [2, 2, 3, 3, 4, 4])


def test_oracle_unimproved_group_is_an_illegal_subscript():
    """A group that never beats best_sigma keeps MATLAB's -1 sentinel.

    COM Octave, bmaxg=0 (no candidate can reduce the norm), N_bg=2 and N_bg=3
    sequential:  "error: bmax(-1): subscripts must be either integers 1 to
    (2^63)-1 or logicals".  Seeding best_ig2 at N_b instead answered with a
    bank that was never scored, and a negative Python slice bound would have
    written it at the far END of bmax.
    """
    for N_bg in (2, 3):
        with pytest.raises(IndexError, match='bmax'):
            floating_taps_1sttest(OCT_HISI, 2, 3, N_bg, 16, 0.0, 0)


def test_oracle_empty_search_range_has_no_best_hcap():
    """When N_bmax-N_bf < N_b+1 the ig1 loop never runs.

    COM Octave, N_b=6 N_bf=3 N_bmax=8: "error: 'best_hcap' undefined near
    line 69" (sequential) and "bmax(-1)" (co-optimised).
    """
    with pytest.raises(ValueError, match='best_hcap'):
        floating_taps_1sttest(OCT_HISI, 6, 3, 1, 8, 0.2, 0)
    with pytest.raises(IndexError, match='bmax'):
        floating_taps_1sttest(OCT_HISI, 6, 3, 1, 8, 0.2, 1)


def test_oracle_nan_in_hisi_selects_nothing():
    """norm() of a vector with a NaN is NaN, and NaN < inf is false.

    COM Octave, hisi with hisi(6)=NaN: "error: 'best_hcap' undefined"
    (sequential) and "bmax(-1)" (co-optimised).  Python had answered with a
    bank at the first candidate position.
    """
    hn = OCT_HISI.copy()
    hn[5] = np.nan
    with pytest.raises(ValueError, match='best_hcap'):
        floating_taps_1sttest(hn, 2, 3, 1, 16, 0.2, 0)
    with pytest.raises(IndexError, match='bmax'):
        floating_taps_1sttest(hn, 2, 3, 1, 16, 0.2, 1)


def test_oracle_hrem_past_the_end_of_hisi():
    """N_bmax beyond length(hisi) walks hrem off the end.

    COM Octave, N_bmax=25 on a 20-tap hisi:
        "error: h(21): out of bound 20 (dimensions are 1x20)".
    """
    with pytest.raises(IndexError):
        floating_taps_1sttest(OCT_HISI, 2, 3, 1, 25, 0.2, 0)
