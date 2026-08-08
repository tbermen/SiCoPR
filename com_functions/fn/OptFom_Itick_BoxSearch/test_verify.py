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
