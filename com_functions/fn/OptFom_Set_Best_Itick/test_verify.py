"""Verification tests for OptFom_Set_Best_Itick().

# ============================================================
# MATLAB GROUND TRUTH (lines 3785-3803)
# if FOM > BEST.itick_FOM: update itick_FOM, itick_in_cluster
# if itick>=0 && FOM > BEST.positive_itick_FOM: update positive fields
# if itick<=0 && FOM > BEST.negative_itick_FOM: update negative fields
# Note: itick=0 triggers BOTH positive and negative branches.
# ============================================================
"""
import math
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Set_Best_Itick.py_impl import OptFom_Set_Best_Itick


def _make(fom, itick):
    THIS = SimpleNamespace(FOM=fom, itick=itick)
    BEST = SimpleNamespace(
        itick_FOM=-math.inf, itick_in_cluster=None,
        positive_itick_FOM=-math.inf, positive_itick_in_loop=None,
        negative_itick_FOM=-math.inf, negative_itick_in_loop=None,
    )
    return THIS, BEST


def test_fom_improvement_updates_cluster():
    THIS, BEST = _make(fom=0.5, itick=2)
    BEST = OptFom_Set_Best_Itick(THIS, BEST)
    assert BEST.itick_FOM == 0.5
    assert BEST.itick_in_cluster == 2


def test_no_improvement_no_update():
    THIS, BEST = _make(fom=0.5, itick=2)
    BEST.itick_FOM = 0.8
    BEST = OptFom_Set_Best_Itick(THIS, BEST)
    assert BEST.itick_FOM == 0.8
    assert BEST.itick_in_cluster is None


def test_positive_itick_updates_positive():
    THIS, BEST = _make(fom=0.5, itick=3)
    BEST = OptFom_Set_Best_Itick(THIS, BEST)
    assert BEST.positive_itick_FOM == 0.5
    assert BEST.positive_itick_in_loop == 3
    assert BEST.negative_itick_FOM == -math.inf


def test_negative_itick_updates_negative():
    THIS, BEST = _make(fom=0.5, itick=-2)
    BEST = OptFom_Set_Best_Itick(THIS, BEST)
    assert BEST.negative_itick_FOM == 0.5
    assert BEST.negative_itick_in_loop == -2
    assert BEST.positive_itick_FOM == -math.inf


def test_zero_itick_updates_both():
    """itick=0 satisfies both itick>=0 and itick<=0."""
    THIS, BEST = _make(fom=0.5, itick=0)
    BEST = OptFom_Set_Best_Itick(THIS, BEST)
    assert BEST.positive_itick_FOM == 0.5
    assert BEST.negative_itick_FOM == 0.5


# ============================================================
# COM Octave oracle — OptFom_Set_Best_Itick extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run by tools/octave_oracle.py.
#
# NO DIVERGENCE FOUND.  16 probes agreed field for field, starting from
# BEST = {itick_FOM: -Inf, itick_in_cluster: -999, positive_itick_FOM: -Inf,
# positive_itick_in_loop: -999, negative_itick_FOM: -Inf,
# negative_itick_in_loop: -999}:
#
#   FOM/itick               itick_FOM  positive        negative
#   0.5 / +2                0.5 @ 2    0.5 @ 2         (untouched)
#   0.5 / -2                0.5 @ -2   (untouched)     0.5 @ -2
#   0.5 / 0                 0.5 @ 0    0.5 @ 0         0.5 @ 0
#   0.5 / +0.5 fractional   0.5 @ 0.5  0.5 @ 0.5       (untouched)
#   0.5 / -0                0.5 @ -0   0.5 @ -0        0.5 @ -0
#   0.5 / NaN               0.5 @ NaN  (untouched)     (untouched)
#   NaN / +2                (untouched, all three)
#   Inf / +2                Inf @ 2    Inf @ 2         (untouched)
#   -Inf / +2 vs -Inf best  (untouched, all three)
#   0.5 / +2 vs NaN best    (untouched, all three)
#
# The comparisons are all STRICT `>`, so an exact tie leaves BEST alone --
# the first candidate to reach a given FOM keeps the slot.  itick is not
# required to be an integer, and -0 satisfies both >=0 and <=0 just as 0
# does.  Every NaN comparison is false, so a NaN FOM never wins and a NaN
# incumbent is never displaced.
#
# One thing an Octave probe cannot see: MATLAB passes BEST by value, so the
# caller's copy is untouched until it assigns the return.  Python's
# SimpleNamespace is mutated in place.  The single call site,
# `BEST = OptFom_Set_Best_Itick(THIS, BEST)` (reference line 9190,
# com_functions/fn/optimize_fom/py_impl.py line 406), reassigns the same
# name and holds no other alias, so the two are equivalent there -- but the
# aliasing is pinned below so a future caller cannot rely on value semantics
# by accident.
# ============================================================

def _best(**over):
    b = SimpleNamespace(
        itick_FOM=-math.inf, itick_in_cluster=-999,
        positive_itick_FOM=-math.inf, positive_itick_in_loop=-999,
        negative_itick_FOM=-math.inf, negative_itick_in_loop=-999)
    for k, v in over.items():
        setattr(b, k, v)
    return b


def _fields(b):
    return (b.itick_FOM, b.itick_in_cluster,
            b.positive_itick_FOM, b.positive_itick_in_loop,
            b.negative_itick_FOM, b.negative_itick_in_loop)


def test_oracle_fresh_updates():
    """The three COM Octave fresh-start cases, field for field."""
    NI = -math.inf
    for itick, expect in [
            (2, (0.5, 2, 0.5, 2, NI, -999)),
            (-2, (0.5, -2, NI, -999, 0.5, -2)),
            (0, (0.5, 0, 0.5, 0, 0.5, 0))]:
        got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=itick),
                                    _best())
        assert _fields(got) == expect


def test_oracle_exact_tie_does_not_displace_the_incumbent():
    """Every comparison is a strict `>`, so an equal FOM changes nothing."""
    b = _best(itick_FOM=0.5, positive_itick_FOM=0.5, negative_itick_FOM=0.5)
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=3), b)
    assert _fields(got) == (0.5, -999, 0.5, -999, 0.5, -999)
    # ... and a tie on positive only still lets the cluster slot update
    b = _best(itick_FOM=0.4, positive_itick_FOM=0.5)
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=3), b)
    assert got.itick_FOM == 0.5 and got.itick_in_cluster == 3
    assert got.positive_itick_FOM == 0.5 and got.positive_itick_in_loop == -999


def test_oracle_nan_never_wins_and_never_loses():
    """NaN comparisons are false both ways."""
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=math.nan, itick=2), _best())
    assert _fields(got) == (-math.inf, -999, -math.inf, -999, -math.inf, -999)
    b = _best(itick_FOM=math.nan, positive_itick_FOM=math.nan,
              negative_itick_FOM=math.nan)
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=2), b)
    assert math.isnan(got.itick_FOM) and got.itick_in_cluster == -999
    assert math.isnan(got.positive_itick_FOM)


def test_oracle_nan_itick_takes_neither_sign_branch():
    """NaN >= 0 and NaN <= 0 are both false, but the cluster slot still wins."""
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=math.nan), _best())
    assert got.itick_FOM == 0.5 and math.isnan(got.itick_in_cluster)
    assert got.positive_itick_in_loop == -999
    assert got.negative_itick_in_loop == -999


def test_oracle_negative_zero_and_fractional_itick():
    """itick is never rounded: -0 takes both branches, 0.5 takes positive."""
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=-0.0), _best())
    assert got.positive_itick_FOM == 0.5 and got.negative_itick_FOM == 0.5
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=0.5), _best())
    assert got.positive_itick_in_loop == 0.5
    assert got.negative_itick_in_loop == -999
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=-0.5), _best())
    assert got.negative_itick_in_loop == -0.5
    assert got.positive_itick_in_loop == -999


def test_oracle_infinite_fom():
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=math.inf, itick=2), _best())
    assert got.itick_FOM == math.inf and got.positive_itick_FOM == math.inf
    # -Inf does not beat an -Inf incumbent
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=-math.inf, itick=2), _best())
    assert got.itick_in_cluster == -999


def test_best_is_mutated_in_place_unlike_matlab():
    """Python aliases BEST where MATLAB copies it; the one call site reassigns."""
    b = _best()
    got = OptFom_Set_Best_Itick(SimpleNamespace(FOM=0.5, itick=2), b)
    assert got is b
    assert b.itick_FOM == 0.5
