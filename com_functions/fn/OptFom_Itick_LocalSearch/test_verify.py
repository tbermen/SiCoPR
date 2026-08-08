"""Verification tests for OptFom_Itick_LocalSearch().

# ============================================================
# MATLAB GROUND TRUTH (lines 3606-3617)
# skip_it = 0
# if middle_search && LocalSearch_Value > 0:
#   if itick>=0 && ~isinf(BEST.positive_itick_FOM) &&
#      |BEST.positive_itick_in_loop - itick| >= LocalSearch_Value: skip_it=1
#   if itick<=0 && ~isinf(BEST.negative_itick_FOM) &&
#      |BEST.negative_itick_in_loop - itick| >= LocalSearch_Value: skip_it=1
# ============================================================
"""
import math
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Itick_LocalSearch.py_impl import OptFom_Itick_LocalSearch


def _best(pos_fom=math.inf, neg_fom=math.inf, pos_loop=0, neg_loop=0):
    return SimpleNamespace(
        positive_itick_FOM=pos_fom,
        negative_itick_FOM=neg_fom,
        positive_itick_in_loop=pos_loop,
        negative_itick_in_loop=neg_loop,
    )


def test_middle_search_false_never_skips():
    """middle_search=False → skip_it=0 always."""
    BEST = _best(pos_fom=1.0, pos_loop=5)
    assert OptFom_Itick_LocalSearch(10, False, BEST, 2) == 0


def test_local_search_value_zero_never_skips():
    """LocalSearch_Value=0 → skip_it=0 always."""
    BEST = _best(pos_fom=1.0, pos_loop=5)
    assert OptFom_Itick_LocalSearch(10, True, BEST, 0) == 0


def test_positive_itick_far_skipped():
    """itick far from best positive → skip_it=1."""
    BEST = _best(pos_fom=1.0, pos_loop=5)
    # |5 - 10| = 5 >= LocalSearch_Value=3
    assert OptFom_Itick_LocalSearch(10, True, BEST, 3) == 1


def test_positive_itick_close_not_skipped():
    """itick close to best positive → skip_it=0."""
    BEST = _best(pos_fom=1.0, pos_loop=5)
    # |5 - 6| = 1 < LocalSearch_Value=3
    assert OptFom_Itick_LocalSearch(6, True, BEST, 3) == 0


def test_positive_fom_inf_not_skipped():
    """positive_itick_FOM=inf → that branch does not skip."""
    BEST = _best(pos_fom=math.inf, pos_loop=5)
    assert OptFom_Itick_LocalSearch(10, True, BEST, 1) == 0


def test_negative_itick_far_skipped():
    """Negative itick far from best negative → skip_it=1."""
    BEST = _best(neg_fom=1.0, neg_loop=-3)
    # |-3 - (-10)| = 7 >= LocalSearch_Value=4
    assert OptFom_Itick_LocalSearch(-10, True, BEST, 4) == 1
