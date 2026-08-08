"""Tests for plot_modal (MATLAB lines ~8914).

MATLAB GROUND TRUTH:
  CM_MASK_REPORT=True, all S-params = 0 → all margins large positive → no fail.
  CM_MASK_REPORT=False → returns None.
  S11 = -40 dB flat → Rlcc margin = mask_value - (-40).
"""
import numpy as np
import pytest
from types import SimpleNamespace
from com_functions.fn.plot_modal.py_impl import plot_modal


def _op(cm_mask=True, show_brd=False):
    return SimpleNamespace(
        CM_MASK_REPORT=cm_mask,
        DISPLAY_WINDOW=False,
        PLOT_CM=False,
        SHOW_BRD=show_brd,
        DEBUG=0,
        TDR=False,
        Report_Modal_ERL='disable',
    )


def _chdata(n_f=200, scc11_val=1e-6):
    f = np.linspace(0.05e9, 67e9, n_f)
    ch = SimpleNamespace()
    ch.faxis = f
    ch.scc11_orig = np.ones(n_f, dtype=complex) * scc11_val
    ch.scd22_orig = np.ones(n_f, dtype=complex) * scc11_val
    ch.sdc22_orig = np.ones(n_f, dtype=complex) * scc11_val
    ch.scc11_raw = ch.scc11_orig
    ch.scd22_raw = ch.scd22_orig
    ch.sdc22_raw = ch.sdc22_orig
    return ch


def test_cm_mask_report_returns_struct():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata()])
    assert rs is not None


def test_no_cm_mask_returns_none():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=False), [_chdata()])
    assert rs is None


def test_required_fields():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata()])
    for f in ('Rlcc_179mm', 'Rlcc_178mm', 'Rlcd_179mm', 'Rldc_179mm',
              'Rlcc_179mm_fail', 'Rlcc_178mm_fail', 'Rlcd_179mm_fail', 'Rldc_179mm_fail'):
        assert hasattr(rs, f), f'missing field: {f}'


def test_small_scc11_no_fail():
    # Very small scc11 → dB is very negative → mask - dB is large positive → no fail
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata(scc11_val=1e-6)])
    assert not rs.Rlcc_179mm_fail
    assert not rs.Rlcc_178mm_fail


def test_fail_detection():
    # Large scc11 → dB ~ 0 → mask (negative) - 0 < 0 → fail
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata(scc11_val=1.0)])
    # At 0 dB, mask is -2 to -4 → margin < 0 → fail expected
    assert rs.Rlcc_179mm_fail or rs.Rlcc_178mm_fail
