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


# ============================================================
# COM Octave oracle values (2026-09-22)
# OptFom_Itick_LocalSearch extracted verbatim from the reference and run under
# Octave on 20 probes.  No divergence was found; these pin the truth table,
# including the cases where isinf() and NaN decide the answer.
#
# Not pinned: middle_search or LocalSearch_Value as [] or a vector.  Octave
# reduces those with all() and answers 0/1, MATLAB's && errors on a non-scalar
# operand, so Octave is not a faithful proxy there.
# ============================================================


def test_oracle_nan_fom_is_not_inf_so_it_skips():
    """~isinf(NaN) is TRUE, so a NaN FOM counts as "a best was found".

    COM Octave, itick=8, best at 0, LS=2, positive_itick_FOM=NaN -> skip 1.
    """
    BEST = _best(pos_fom=math.nan, pos_loop=0)
    assert OptFom_Itick_LocalSearch(8, 1, BEST, 2) == 1


def test_oracle_nan_in_loop_never_reaches_the_threshold():
    """abs(NaN-itick) >= LS is false, so a NaN location never skips.

    COM Octave, itick=8, positive_itick_in_loop=NaN, LS=2 -> skip 0.
    """
    BEST = _best(pos_fom=1.5, pos_loop=math.nan)
    assert OptFom_Itick_LocalSearch(8, 1, BEST, 2) == 0


def test_oracle_nan_local_search_value_is_not_positive():
    """COM Octave, LocalSearch_Value=NaN -> skip 0 (NaN>0 is false)."""
    BEST = _best(pos_fom=1.5, pos_loop=0)
    assert OptFom_Itick_LocalSearch(8, 1, BEST, math.nan) == 0


def test_oracle_negative_local_search_value_never_skips():
    """COM Octave, LocalSearch_Value=-2 -> skip 0."""
    BEST = _best(pos_fom=1.5, pos_loop=0)
    assert OptFom_Itick_LocalSearch(8, 1, BEST, -2) == 0


def test_oracle_distance_exactly_equal_to_the_step_skips():
    """The test is >=, not >.  COM Octave, itick=2, best 0, LS=2 -> skip 1."""
    BEST = _best(pos_fom=1.5, pos_loop=0)
    assert OptFom_Itick_LocalSearch(2, 1, BEST, 2) == 1


def test_oracle_itick_zero_takes_both_branches():
    """itick=0 satisfies both >=0 and <=0, so either bank can skip it.

    COM Octave, LS=2: only the negative bank populated (FOM 1.5 at 9) -> 1;
    only the positive bank populated (FOM 1.5 at 9) -> 1.
    """
    assert OptFom_Itick_LocalSearch(0, 1, _best(neg_fom=1.5, neg_loop=9), 2) == 1
    assert OptFom_Itick_LocalSearch(0, 1, _best(pos_fom=1.5, pos_loop=9), 2) == 1


def test_oracle_minus_inf_fom_means_nothing_found_yet():
    """COM Octave, both FOMs -Inf, itick=4, LS=2 -> skip 0."""
    assert OptFom_Itick_LocalSearch(4, 1, _best(-math.inf, -math.inf), 2) == 0


def test_oracle_unset_opposite_bank_is_never_read():
    """A positive itick short-circuits before the negative fields.

    COM Octave with only the positive fields set on BEST, itick=8, LS=2:
    skip 1, no error about the missing negative_itick_FOM.
    """
    BEST = SimpleNamespace(positive_itick_FOM=1.5, positive_itick_in_loop=0)
    assert OptFom_Itick_LocalSearch(8, 1, BEST, 2) == 1
