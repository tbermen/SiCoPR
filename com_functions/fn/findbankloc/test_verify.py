"""Verification tests for findbankloc().

# ============================================================
# MATLAB GROUND TRUTH
# Returns indices of N_bg groups of tap_bk taps that maximise
# sum(h0^2 - h1^2) where h1 = max(0, h0 - bmaxg*curval)
# Returned indices are 0-based (Python convention; MATLAB returned 1-based).
#
# Simple case: hisi=[0,0,1,0,0], N_bg=1, tap_bk=1, idx_st=1, idx_en=5
#   h0=[0,0,1,0,0]; h1=max(0,h0-bmaxg*curval); ndiff[j]=h0[j]^2-h1[j]^2
#   max ndiff at j=2 (0-based) → returned 0-based hisi idx = 2
#
# idx_st/idx_en are 1-based MATLAB conventions
# returned array length = N_bg * tap_bk
# all returned indices in range [idx_st-1, idx_en-1]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.findbankloc.py_impl import findbankloc


def test_single_peak_single_tap():
    """hisi with clear peak at index 2: best single tap should select it."""
    hisi = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    idx = findbankloc(hisi, 1, 5, 1, 0.5, 1.0, 1)
    assert len(idx) == 1
    assert idx[0] == 2   # 0-based Python index


def test_result_length():
    """Result has N_bg * tap_bk elements.

    Seeded on purpose. This used an unseeded np.random.rand(20) and so was a
    4.4% chance of a red run: on those draws the strongest bank start landed
    within tap_bk-1 of the end of ndiff and the function raised IndexError.
    It failed the gate once, passed on the next three runs, and was only
    reproducible by sweeping seeds. An unseeded input is not extra coverage --
    it is a test that reports a different verdict each time it is asked.
    """
    hisi = np.random.RandomState(0).rand(20)
    idx = findbankloc(hisi, 1, 20, 2, 0.5, 0.8, 2)
    assert len(idx) == 4


def test_bank_start_near_the_end_does_not_raise():
    """The overshoot case that made the seeded test above necessary.

    ndiff is indexed by bank START position, so it is tap_bk-1 shorter than h0.
    When the strongest start is one of those last tap_bk-1 positions, MATLAB's
    `ndiff(new_bank)=min_energy` grows the array; NumPy raised IndexError.

    Seed 29 is the first of 3000 that hits it. The sweep is kept small but wide
    enough that a regression shows up as more than one failure.
    """
    for seed in (29, 51, 63):
        hisi = np.random.RandomState(seed).rand(20)
        idx = findbankloc(hisi, 1, 20, 2, 0.5, 0.8, 2)
        assert len(idx) == 4, 'seed %d returned %d indices' % (seed, len(idx))
        assert idx.min() >= 0 and idx.max() <= 19


# A hand-built input was tried here -- hisi peaking on its last two samples --
# on the assumption that it would force a bank start into the overshoot window.
# It does not: the goodV / set_next_bank branch shifts the chosen bank one
# position earlier, so the write stays in range and the test passed with the bug
# reintroduced. It is left out rather than left in looking like coverage.
# test_bank_start_near_the_end_does_not_raise is the one that actually fails
# when the fix is reverted, confirmed by mutation.


def test_indices_in_range():
    """All returned indices within [idx_st-1, idx_en-1]."""
    n = 15
    hisi = np.random.rand(n)
    idx = findbankloc(hisi, 3, 12, 2, 0.5, 1.0, 1)
    assert np.all(idx >= 2)   # idx_st-1 = 3-1 = 2
    assert np.all(idx <= 11)  # idx_en-1 = 12-1 = 11


def test_two_groups_non_overlapping():
    """Two groups of tap_bk=1 should pick two different indices."""
    hisi = np.array([0.0, 0.9, 0.0, 0.8, 0.0, 0.0])
    idx = findbankloc(hisi, 1, 6, 1, 0.5, 2.0, 2)
    assert len(idx) == 2
    assert idx[0] != idx[1]


def test_curval_negative_sets_h1_zero():
    """curval<0: h1=0, ndiff=h0^2 everywhere; function still returns valid indices."""
    hisi = np.array([0.1, 0.5, 0.2, 0.3, 0.1])
    idx = findbankloc(hisi, 1, 5, 1, -0.5, 1.0, 1)
    assert len(idx) == 1
    assert 0 <= idx[0] <= 4


# ---------------------------------------------------------------------------
# Against COM Octave. findbankloc returns tap-bank locations; MATLAB indexes
# them from 1 and SiCoPR from 0, so every location here is the reference's
# minus one. Pinning the relationship means a convention change cannot pass.
# ---------------------------------------------------------------------------

_OCT_IDX_1BASED = [1, 2, 8, 9]


def _isi():
    """A decaying ISI tail with bumps at 7-8 and 14, so the chosen banks have
    to respond to the input rather than fall on the first locations."""
    rng = np.random.default_rng(3)
    h = 0.3 * np.exp(-np.arange(20) / 6.0) + 0.02 * rng.standard_normal(20)
    h[7] += 0.20
    h[8] += 0.15
    h[14] += 0.12
    return h


def test_matches_com_octave_one_based_minus_one():
    got = np.asarray(findbankloc(_isi(), 1, 20, 2, np.inf, 0.2, 2)).ravel()
    want = [v - 1 for v in _OCT_IDX_1BASED]
    assert list(got) == want, (
        'locations %r; COM Octave gives %r 1-based, so %r is expected here'
        % (list(got), _OCT_IDX_1BASED, want))


def test_locations_follow_the_isi_bumps():
    """Move the energy and the banks must move with it."""
    h = _isi()
    moved = h.copy()
    moved[7] -= 0.20
    moved[8] -= 0.15
    moved[15] += 0.30
    a = list(np.asarray(findbankloc(h, 1, 20, 2, np.inf, 0.2, 2)).ravel())
    b = list(np.asarray(findbankloc(moved, 1, 20, 2, np.inf, 0.2, 2)).ravel())
    assert a != b, 'the same banks %r were chosen for a different ISI tail' % a
