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


# ============================================================
# COM Octave oracle values — max() and the dfetaps colon evaluated under Octave
# on the same operands.  Pinned 2026-09-22.
#
# Two divergences these pin:
#  * the empty-cursor fallback `[~,THIS.cursor_i]=max(sbr)`.  MATLAB's max
#    SKIPS NaN unless every element is NaN; np.argmax returns the index of the
#    first NaN, so a single bad sample put the cursor at the head of the pulse
#    response.
#  * `sbr(cursor_i+M : M : cursor_i+M*ndfe)`.  MATLAB refuses to read past the
#    end; a python slice stops quietly, so a short response yielded fewer than
#    param.ndfe taps with no complaint.
#
# COM Octave:
#   [m,i] = max([NaN 1 3 2]) -> m=3,   i=3   (np.argmax gave 0)
#   [m,i] = max([1 NaN 5])   -> m=5,   i=3   (np.argmax gave 1)
#   [m,i] = max([NaN NaN])   -> m=NaN, i=1   (np.argmax gave 0)
#   sbr=1:20; sbr(5+4*(1):4:5+4*3)/sbr(5)
#     -> [1.8, 2.6000000000000001, 3.3999999999999999]
#   sbr=1:20; sbr(5+4*(1):4:5+4*6)/sbr(5)
#     -> "error: sbr(29): out of bound 20 (dimensions are 1x20)"
# ============================================================

@pytest.mark.parametrize('head,all_nan,expected_0based', [
    ([np.nan, 1.0, 3.0, 2.0], False, 2),
    ([1.0, np.nan, 5.0], False, 2),
    ([], True, 0),
])
def test_octave_empty_cursor_fallback_skips_nan(head, all_nan, expected_0based):
    BEST = SimpleNamespace()
    sbr = np.full(200, np.nan) if all_nan else np.zeros(200)
    if head:
        sbr[:len(head)] = head
    param = _param(M=8, ndfe=3)
    with np.errstate(invalid='ignore'):
        result = OptFom_Update_Best_Settings_EQ_Failed(
            BEST, _THIS(cursor_i=None), sbr, _chdata(), param, _op())
    assert result.cursor_i == expected_0based


def test_octave_dfetaps_in_range_values():
    """COM Octave: sbr=1:20, cursor_i=5 (1-based), M=4, ndfe=3 ->
    [1.8, 2.6000000000000001, 3.3999999999999999]"""
    BEST = SimpleNamespace()
    sbr = np.arange(1.0, 21.0)
    param = _param(M=4, ndfe=3)
    result = OptFom_Update_Best_Settings_EQ_Failed(
        BEST, _THIS(cursor_i=4), sbr, _chdata(), param, _op())
    np.testing.assert_array_equal(
        result.dfetaps,
        np.array([1.8, 2.6000000000000001, 3.3999999999999999]))


def test_octave_dfetaps_past_end_raises():
    """COM Octave: same sbr with ndfe=6 ->
    "error: sbr(29): out of bound 20 (dimensions are 1x20)".
    The port returned 3 taps instead of 6."""
    BEST = SimpleNamespace()
    sbr = np.arange(1.0, 21.0)
    param = _param(M=4, ndfe=6)
    with pytest.raises(IndexError):
        OptFom_Update_Best_Settings_EQ_Failed(
            BEST, _THIS(cursor_i=4), sbr, _chdata(), param, _op())
