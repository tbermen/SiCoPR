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
