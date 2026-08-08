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


def test_n_bg_zero_returns_zero_bmax():
    """N_bg=0: early return with bmax=0."""
    hisi = np.array([0.1, 0.5, 0.3, 0.2])
    bmax, locs = floating_taps_1sttest(hisi, 1, 1, 0, 4, 0.5)
    assert np.all(bmax == 0)
    assert len(locs) == 0


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
