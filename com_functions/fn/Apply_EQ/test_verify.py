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


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-22): Apply_EQ run verbatim under Octave from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m for this function).
#
# Two divergences it found:
#
# 1. The pulse-response extension built the new time stamps as
#    t(end) + k*(1/(fb*M)).  MATLAB L927 is (1:samples_added)/fb/M + t(end) --
#    two successive divisions, not one multiply by a pre-formed dt.  They are
#    different doubles (2 of 7, and 3 of 13, sample offsets differ at
#    fb=25e9 M=8), and the last extended time stamp came out
#    2.2999999999999998e-10 where Octave gives 2.3000000000000001e-10.
#
# 2. param.CTLE_type outside {CL93, CL120d, CL120e} fell through to the
#    unequalised response.  MATLAB's switch has no `otherwise`, so eq_ir is
#    never assigned:  COM Octave, CTLE_type='NOSUCH', INCLUDE_CTLE=1 ->
#    "error: 'eq_ir' undefined near line 46, column 31".
#
# Convention: fom_result.t_s is 0-BASED here (it is BEST.cursor_i) and 1-based
# in MATLAB, so the oracle run below used fom_result.t_s = 4 for t_s = 3.
# ---------------------------------------------------------------------------
_OCT_M, _OCT_FB, _OCT_N, _OCT_SBR = 8, 25e9, 40, 47


def _oct_ir():
    ir = np.zeros(_OCT_N)
    ir[5] = 1.0
    ir[6] = -0.3
    ir[9] = 0.12
    return ir


def _oct_inputs(ctle_type='CL93', include_ctle=0):
    ir = _oct_ir()
    p = SimpleNamespace(
        fb=_OCT_FB, CTLE_fz=np.array([5e9, 6e9]), CTLE_fp1=np.array([10e9, 11e9]),
        CTLE_fp2=np.array([20e9, 21e9]), ctle_gdc_values=np.array([0.0, -3.0]),
        CTLE_type=ctle_type, samples_per_ui=_OCT_M, number_of_s4p_files=1,
        f_HP=None, g_DC_HP_values=None, f_HP_Z=None, f_HP_P=None)
    fom = SimpleNamespace(ctle=1, best_G_high_pass=1, sbr=np.zeros(_OCT_SBR),
                          txffe=np.array([0.0, -0.2, 0.9, -0.1, 0.0]), cur=3,
                          t_s=3, RxFFE=np.array([0.0, 1.0, 0.0]))
    cd = [SimpleNamespace(uneq_imp_response=ir.copy(),
                          uneq_pulse_response=np.cumsum(ir),
                          t=np.arange(_OCT_N) / (_OCT_FB * _OCT_M), type='THRU')]
    op = SimpleNamespace(INCLUDE_CTLE=include_ctle, RxFFE=False,
                         FFE_OPT_METHOD='OTHER')
    return p, fom, cd, op


def test_octave_extended_time_axis_is_two_divisions():
    """COM Octave: the 7 samples appended to chdata(1).t.

    Bit-exact.  t(end)+k*(1/(fb*M)) puts the last stamp at
    2.2999999999999998e-10; MATLAB's (1:k)/fb/M + t(end) gives
    2.3000000000000001e-10."""
    p, fom, cd, op = _oct_inputs()
    out = Apply_EQ(p, fom, cd, op)
    t = np.asarray(out[0].t)
    assert t.size == _OCT_SBR
    expect_tail = np.array([2.0000000000000001e-10, 2.0499999999999999e-10,
                            2.1e-10, 2.1500000000000001e-10,
                            2.1999999999999999e-10, 2.25e-10,
                            2.3000000000000001e-10])
    np.testing.assert_array_equal(t[_OCT_N:], expect_tail)


def test_octave_eq_pulse_response_with_txffe():
    """COM Octave, INCLUDE_CTLE=0, txffe=[0 -0.2 0.9 -0.1 0], cur=3, M=8."""
    p, fom, cd, op = _oct_inputs()
    out = Apply_EQ(p, fom, cd, op)
    expect = np.array([
        -0.13999999999999999, -0.16400000000000001, -0.16400000000000001,
        -0.16400000000000001, -0.16400000000000001, 0.93600000000000005,
        0.60599999999999998, 0.60599999999999998, 0.60599999999999998,
        0.73799999999999999, 0.73799999999999999, 0.73799999999999999,
        0.73799999999999999, -0.26200000000000001, 0.038000000000000006,
        0.038000000000000006, 0.038000000000000006, -0.082000000000000003,
        -0.082000000000000003, -0.082000000000000003, -0.082000000000000003,
        0.017999999999999999, -0.012, -0.012, -0.012, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.20000000000000001,
        -0.13999999999999999, -0.13999999999999999])
    np.testing.assert_array_equal(np.asarray(out[0].eq_pulse_response), expect)


def test_octave_sampled_pulse_and_time():
    """COM Octave, sample_start = mod(t_s-1, M)+1 with MATLAB t_s = 4."""
    p, fom, cd, op = _oct_inputs()
    out = Apply_EQ(p, fom, cd, op)
    np.testing.assert_array_equal(
        np.asarray(out[0].pulse_sampled_w_tx_ffe_ctle),
        np.array([-0.16400000000000001, 0.73799999999999999,
                  -0.082000000000000003, 0, 0, 0]))
    np.testing.assert_array_equal(
        np.asarray(out[0].t_sampled_w_tx_ffe_ctle),
        np.array([1.5e-11, 5.4999999999999997e-11, 9.4999999999999995e-11,
                  1.35e-10, 1.7499999999999999e-10, 2.1500000000000001e-10]))


def test_octave_unknown_ctle_type_raises():
    """COM Octave, CTLE_type='NOSUCH' with INCLUDE_CTLE=1:
    "error: 'eq_ir' undefined".  The port returned the unequalised channel."""
    p, fom, cd, op = _oct_inputs(ctle_type='NOSUCH', include_ctle=1)
    with pytest.raises(ValueError, match='CTLE_type'):
        Apply_EQ(p, fom, cd, op)
