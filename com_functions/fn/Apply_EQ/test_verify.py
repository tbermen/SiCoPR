"""Verification tests for Apply_EQ().

# ============================================================
# MATLAB GROUND TRUTH (lines 905-976)
# Applies CTLE, TX-FFE to each channel's impulse/pulse response.
# INCLUDE_CTLE=0: eq_ir = uneq_ir (passthrough).
# CTLE_type='CL93': single TD_CTLE call.
# FFE applied for THRU/FEXT types.
# OP.RxFFE=False: eq_pulse_response set without force().
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com_functions.fn.Apply_EQ.py_impl as _mod
from com_functions.fn.Apply_EQ.py_impl import Apply_EQ


def _ir(N=200, peak=10):
    ir = np.zeros(N)
    ir[peak] = 1.0
    return ir


def _param(M=8, n_files=1):
    return SimpleNamespace(
        fb=25e9,
        CTLE_fz=np.array([5e9]),
        CTLE_fp1=np.array([10e9]),
        CTLE_fp2=np.array([20e9]),
        ctle_gdc_values=np.array([0.0]),
        CTLE_type='CL93',
        samples_per_ui=M,
        number_of_s4p_files=n_files,
        f_HP=None,
        g_DC_HP_values=None,
        f_HP_Z=None,
        f_HP_P=None,
    )


def _fom(M=8, t_s=1, cur=5):
    ir = _ir()
    return SimpleNamespace(
        ctle=1,
        best_G_high_pass=1,
        sbr=ir,
        txffe=np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        cur=cur,
        t_s=t_s,
        RxFFE=np.array([0.0, 1.0, 0.0]),
    )


def _op():
    return SimpleNamespace(INCLUDE_CTLE=0, RxFFE=False, FFE_OPT_METHOD='OTHER')


def _chdata(N=200, ctype='THRU', M=8):
    ir = _ir(N)
    t = np.arange(N) / (25e9 * M)
    return [SimpleNamespace(
        uneq_imp_response=ir,
        uneq_pulse_response=np.cumsum(ir),
        t=t,
        type=ctype,
    )]


def test_passthrough_no_ctle():
    """INCLUDE_CTLE=0: eq_imp_response == uneq_imp_response."""
    p = _param()
    fom = _fom()
    cd = _chdata()
    result = Apply_EQ(p, fom, cd, _op())
    np.testing.assert_array_equal(result[0].eq_imp_response, cd[0].uneq_imp_response)


def test_eq_pulse_response_set():
    """eq_pulse_response is set when RxFFE=False."""
    p = _param()
    fom = _fom()
    cd = _chdata()
    result = Apply_EQ(p, fom, cd, _op())
    assert hasattr(result[0], 'eq_pulse_response')


def test_pulse_sampled_set():
    """pulse_sampled_w_tx_ffe_ctle is a downsampled version of eq_pulse."""
    p = _param(M=8)
    fom = _fom(M=8, t_s=1)
    cd = _chdata(M=8)
    result = Apply_EQ(p, fom, cd, _op())
    assert hasattr(result[0], 'pulse_sampled_w_tx_ffe_ctle')
    assert len(result[0].pulse_sampled_w_tx_ffe_ctle) > 0


def test_ctle_applied_cl93():
    """INCLUDE_CTLE=1, CL93: eq_imp_response != uneq_imp_response."""
    p = _param()
    fom = _fom()
    cd = _chdata()
    op = SimpleNamespace(INCLUDE_CTLE=1, RxFFE=False, FFE_OPT_METHOD='OTHER')
    result = Apply_EQ(p, fom, cd, op)
    assert not np.allclose(result[0].eq_imp_response, cd[0].uneq_imp_response)


def test_rxffe_invokes_force(monkeypatch):
    """RxFFE=True routes the pulse through force() (no longer a stub).

    force() is a top-level fn in the assembled module; inject a spy to confirm
    the RxFFE branch reaches it and eq_pulse_response is still set. (force() is
    unit-tested in its own directory.)"""
    p = _param()
    fom = _fom()
    cd = _chdata()
    calls = {'n': 0}

    def spy_force(eq_pulse, param, OP, t_s, RxFFE):
        calls['n'] += 1
        return np.asarray(eq_pulse), None, None

    monkeypatch.setattr(_mod, 'force', spy_force, raising=False)
    op = SimpleNamespace(INCLUDE_CTLE=0, RxFFE=True, FFE_OPT_METHOD='OTHER')
    result = Apply_EQ(p, fom, cd, op)
    assert calls['n'] == 1
    assert hasattr(result[0], 'eq_pulse_response')
