"""Verification tests for OptFom_Create_Output().

# ============================================================
# MATLAB GROUND TRUTH (lines 3418-3504)
# Fills result struct from BEST and chdata.
# A_f: sum(PR_window) / M  (eq 163A-3).
# Tr_measured_from_step: (i80-i20)/(fb*M).
# DFE_taps_i: cursor_i + arange(1, ndfe+1)*M.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Create_Output.py_impl import OptFom_Create_Output


def _param(M=8, fb=25e9, ndfe=3, N_v=10, D_p=3):
    return SimpleNamespace(
        samples_per_ui=M, fb=fb, ndfe=ndfe,
        N_v=N_v, D_p=D_p,
        cursor_index=1,
        Floating_DFE=False, Floating_RXFFE=False,
    )


def _BEST(cursor_i=40, M=8, ndfe=3, N=200):
    sbr = np.zeros(N)
    sbr[cursor_i] = 1.0
    return SimpleNamespace(
        cursor_i=cursor_i,
        txffe=np.array([0.0, 1.0, 0.0]),
        ctle=1,
        G_high_pass=1,
        dfetaps=np.zeros(ndfe),
        A_s=0.5,
        itick=1,
        sigma_N=0.01,
        h_J=np.zeros(10),
        FOM=10.0,
        sbr=sbr,
        A_p=1.0,
        ISI=0.05,
        ffegain=1.0,
        bmax=np.ones(ndfe) * 0.5,
        bmin=-np.ones(ndfe) * 0.5,
        tail_RSS=0.0,
        sampled_sbr_precursors_t=[],
        sampled_sbr_postcursors_t=[],
    )


def _chdata(M=8, fb=25e9, N=200):
    ir = np.zeros(N)
    ir[50] = 1.0
    ir[51] = 0.5
    return [SimpleNamespace(uneq_pulse_response=ir.copy())]


def _op():
    return SimpleNamespace(TDMODE=True, RxFFE=False)


def test_returns_result():
    """OptFom_Create_Output returns the result object."""
    result = SimpleNamespace()
    out = OptFom_Create_Output(result, _BEST(), 0.0, _chdata(), _param(), _op())
    assert hasattr(out, 'A_s')


def test_DFE_taps_i_shape():
    """DFE_taps_i has ndfe elements."""
    result = SimpleNamespace()
    out = OptFom_Create_Output(result, _BEST(), 0.0, _chdata(), _param(ndfe=3), _op())
    assert len(out.DFE_taps_i) == 3


def test_A_f_positive():
    """A_f (steady-state voltage) is positive for non-trivial PR."""
    result = SimpleNamespace()
    out = OptFom_Create_Output(result, _BEST(), 0.0, _chdata(), _param(), _op())
    assert out.A_f > 0


def test_Tr_measured_non_negative():
    """Tr_measured_from_step is non-negative."""
    result = SimpleNamespace()
    out = OptFom_Create_Output(result, _BEST(), 0.0, _chdata(), _param(), _op())
    assert out.Tr_measured_from_step >= 0


def test_txffe_assigned():
    """result.txffe equals BEST.txffe."""
    result = SimpleNamespace()
    BEST = _BEST()
    out = OptFom_Create_Output(result, BEST, 0.0, _chdata(), _param(), _op())
    assert np.allclose(out.txffe, BEST.txffe)


def test_cursor_assigned():
    """result.t_s equals BEST.cursor_i."""
    result = SimpleNamespace()
    BEST = _BEST(cursor_i=40)
    out = OptFom_Create_Output(result, BEST, 0.0, _chdata(), _param(), _op())
    assert out.t_s == 40
