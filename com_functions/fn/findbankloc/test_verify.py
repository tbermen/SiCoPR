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
    """Result has N_bg * tap_bk elements."""
    hisi = np.random.rand(20)
    idx = findbankloc(hisi, 1, 20, 2, 0.5, 0.8, 2)
    assert len(idx) == 4


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
