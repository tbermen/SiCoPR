"""Verification tests for OptFom_Update_Best_Settings_EQ_Failed().

# ============================================================
# MATLAB GROUND TRUTH (lines 3891-3934)
# Sets BEST fields when EQ optimization fails.
# If THIS.cursor_i is empty → use argmax(sbr) as cursor.
# BEST.dfetaps = UI-spaced postcursors / cursor value.
# ============================================================
"""
import numpy as np
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Update_Best_Settings_EQ_Failed.py_impl import OptFom_Update_Best_Settings_EQ_Failed


def _sbr(M=8, peak=40, N=200):
    sbr = np.zeros(N)
    sbr[peak] = 1.0
    return sbr


def _THIS(cursor_i=40):
    return SimpleNamespace(
        txffe=[0, 0, 1, 0, 0],
        ctle_index=1,
        g_LP_index=0,
        FOM=-25.0,
        cursor_i=cursor_i,
        itick=5,
        sigma_N=0.01,
        h_J=0.001,
    )


def _param(M=8, ndfe=3):
    return SimpleNamespace(
        cursor_index=5,
        samples_per_ui=M,
        ndfe=ndfe,
        bmax=[1.0],
        bmin=[-1.0],
        Floating_DFE=False,
    )


def _op():
    return SimpleNamespace(TDMODE=False, RxFFE=False)


def _chdata(M=8):
    ir = np.zeros(200)
    ir[40] = 1.0
    return [SimpleNamespace(ctle_imp_response=ir)]


def test_fom_set():
    """BEST.FOM is updated."""
    BEST = SimpleNamespace()
    sbr = _sbr()
    result = OptFom_Update_Best_Settings_EQ_Failed(BEST, _THIS(), sbr, _chdata(), _param(), _op())
    assert result.FOM == _THIS().FOM


def test_empty_cursor_uses_argmax():
    """When cursor_i is None, uses argmax(sbr) as cursor."""
    BEST = SimpleNamespace()
    sbr = _sbr(peak=40)
    result = OptFom_Update_Best_Settings_EQ_Failed(BEST, _THIS(cursor_i=None), sbr, _chdata(), _param(), _op())
    assert result.cursor_i == 40


def test_a_p_is_max_sbr():
    """BEST.A_p = max(sbr)."""
    BEST = SimpleNamespace()
    sbr = _sbr()
    result = OptFom_Update_Best_Settings_EQ_Failed(BEST, _THIS(), sbr, _chdata(), _param(), _op())
    assert result.A_p == pytest.approx(float(np.max(sbr)))


def test_dfetaps_computed():
    """BEST.dfetaps is set and has ndfe elements."""
    BEST = SimpleNamespace()
    sbr = _sbr(M=8, peak=20, N=200)
    sbr[28] = 0.3  # cursor+M (postcursor 1)
    sbr[36] = 0.2  # cursor+2M (postcursor 2)
    sbr[44] = 0.1  # cursor+3M (postcursor 3)
    result = OptFom_Update_Best_Settings_EQ_Failed(BEST, _THIS(cursor_i=20), sbr, _chdata(), _param(M=8, ndfe=3), _op())
    assert len(result.dfetaps) == 3


import pytest
def test_a_s_equals_sbr_at_cursor():
    """BEST.A_s = sbr[cursor_i]."""
    BEST = SimpleNamespace()
    sbr = _sbr(peak=40)
    result = OptFom_Update_Best_Settings_EQ_Failed(BEST, _THIS(cursor_i=40), sbr, _chdata(), _param(), _op())
    assert result.A_s == pytest.approx(float(sbr[40]))
