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


# ============================================================
# COM Octave 4p16p0 oracle pins.
# OptFom_Create_Output extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave on the fixtures above:
# M=8, fb=25e9, ndfe=3, N_v=10, D_p=3, cursor_index=1; BEST.cursor_i=40
# (Octave 41), A_p=1, ISI=0.05; chdata(1).uneq_pulse_response 200 samples
# with 1.0 at index 50 and 0.5 at index 51 (Octave 51 and 52).
#
# result.DFE_taps_i is deliberately NOT pinned against Octave: the port keeps
# cursor_i 0-based, so its DFE_taps_i is Octave's minus one throughout.
# ============================================================
_OCT_A_F_IMPULSE = 0.1875
_OCT_PMAX_BY_VF_IMPULSE = 5.333333333333333
_OCT_SNR_ISI = 26.020599913279625


def _BEST_full(**kw):
    """_BEST plus the fields only the RxFFE and floating-tap branches read."""
    b = _BEST()
    b.RxFFE = np.array([0.1, 1.0, -0.2])
    b.PSD_results = SimpleNamespace(S_n_rms=7)
    b.MMSE_results = SimpleNamespace(x=5)
    b.floating_tap_locations = np.array([1.0, 2.0])
    b.floating_tap_coef = np.array([0.01, 0.02])
    b.IR = np.zeros(100)
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def _pulse_chdata(N=200):
    """A pulse wide enough that the 20% and 80% step crossings differ."""
    n = np.arange(N)
    pr = (np.exp(-((n - 50.0) / 5.0) ** 2)
          + 0.25 * np.exp(-((n - 58.0) / 7.0) ** 2))
    return [SimpleNamespace(uneq_pulse_response=pr)]


def test_octave_A_p_A_f_and_pmax_by_vf():
    """COM Octave: A_p = max(PR), A_f = sum(PR window)/M, Pmax_by_Vf their ratio."""
    out = OptFom_Create_Output(SimpleNamespace(), _BEST(), 0.0, _chdata(),
                               _param(), _op())
    assert out.A_p == pytest.approx(1.0, rel=1e-15)
    assert out.A_f == pytest.approx(_OCT_A_F_IMPULSE, rel=1e-15)
    assert out.Pmax_by_Vf == pytest.approx(_OCT_PMAX_BY_VF_IMPULSE, rel=1e-14)
    assert out.SNR_ISI == pytest.approx(_OCT_SNR_ISI, rel=1e-14)
    assert out.Tr_measured_from_step == 0.0


def test_octave_rise_time_from_the_step_response():
    """COM Octave, a pulse whose step response crosses 20% and 80% apart:
    A_f = 1.495507936701499, Tr = 3.5000000000000002e-11 s (7 samples at
    fb*M), Pmax_by_Vf = 0.71394945885247452."""
    out = OptFom_Create_Output(SimpleNamespace(), _BEST(), 0.0, _pulse_chdata(),
                               _param(), _op())
    assert out.A_p == pytest.approx(1.0677170821176158, rel=1e-14)
    assert out.A_f == pytest.approx(1.495507936701499, rel=1e-14)
    assert out.Tr_measured_from_step == pytest.approx(3.5000000000000002e-11,
                                                      rel=1e-13)
    assert out.Pmax_by_Vf == pytest.approx(0.71394945885247452, rel=1e-14)


def test_octave_window_clamps_at_both_ends():
    """COM Octave: ibeg clamps to 1 and iend to length(PR), and either way the
    window still holds the whole pulse, so A_f = 1/M."""
    early = np.zeros(200)
    early[1] = 1.0
    out = OptFom_Create_Output(SimpleNamespace(), _BEST(), 0.0,
                               [SimpleNamespace(uneq_pulse_response=early)],
                               _param(), _op())
    assert out.A_f == pytest.approx(0.125, rel=1e-15)
    late = np.zeros(200)
    late[195] = 1.0
    out = OptFom_Create_Output(SimpleNamespace(), _BEST(), 0.0,
                               [SimpleNamespace(uneq_pulse_response=late)],
                               _param(), _op())
    assert out.A_f == pytest.approx(0.125, rel=1e-15)


def test_octave_pmax_by_vf_is_infinite_when_A_f_is_zero():
    """COM Octave: A_p/A_f divides straight through, so a PR window summing to
    zero gives Inf.  Substituting 0.0 hid a degenerate channel."""
    pr = np.zeros(200)
    pr[50] = 1.0
    pr[58] = -1.0
    out = OptFom_Create_Output(SimpleNamespace(), _BEST(), 0.0,
                               [SimpleNamespace(uneq_pulse_response=pr)],
                               _param(), _op())
    assert out.A_f == 0.0
    assert np.isinf(out.Pmax_by_Vf) and out.Pmax_by_Vf > 0


def test_octave_snr_isi_zero_over_zero_is_nan_not_inf():
    """COM Octave: 20*log10(0/0) is NaN.  A_p=1, ISI=0 is Inf, and both are
    reached by dividing rather than by testing ISI against zero."""
    out = OptFom_Create_Output(SimpleNamespace(), _BEST_full(ISI=0.0), 0.0,
                               _chdata(), _param(), _op())
    assert np.isinf(out.SNR_ISI)
    out = OptFom_Create_Output(SimpleNamespace(),
                               _BEST_full(ISI=0.0, A_p=0.0), 0.0, _chdata(),
                               _param(), _op())
    assert np.isnan(out.SNR_ISI)


def test_octave_snr_isi_of_a_negative_ratio_is_complex():
    """COM Octave, BEST.A_p = -1 with ISI = 0.05:
    SNR_ISI = 26.020599913279625 + 27.287527076836827i.
    MATLAB log10 of a negative argument is complex; np.log10 gives NaN."""
    out = OptFom_Create_Output(SimpleNamespace(), _BEST_full(A_p=-1.0), 0.0,
                               _chdata(), _param(), _op())
    assert out.SNR_ISI == pytest.approx(
        complex(26.020599913279625, 27.287527076836827), rel=1e-14)


def test_octave_ffe_opt_method_strcmp_is_case_sensitive():
    """COM Octave, OP.RxFFE=1: only the exact string 'MMSE' adds
    result.PSD_results and result.MMSE_results.  'mmse' and 'Mmse' add
    neither; .upper() here attached both on every spelling."""
    op = _op()
    op.RxFFE = True
    for method, expected in (('MMSE', True), ('mmse', False),
                             ('Mmse', False), ('MMSE ', False)):
        op.FFE_OPT_METHOD = method
        out = OptFom_Create_Output(SimpleNamespace(), _BEST_full(), 0.0,
                                   _chdata(), _param(), op)
        assert hasattr(out, 'RxFFE')
        assert hasattr(out, 'PSD_results') is expected, method
        assert hasattr(out, 'MMSE_results') is expected, method


def test_octave_floating_tap_fields_follow_the_two_switches():
    """COM Octave: Floating_DFE adds locations and coefficients,
    Floating_RXFFE adds locations only."""
    p = _param()
    p.Floating_DFE = True
    out = OptFom_Create_Output(SimpleNamespace(), _BEST_full(), 0.0, _chdata(),
                               p, _op())
    assert hasattr(out, 'floating_tap_locations')
    assert hasattr(out, 'floating_tap_coef')
    p = _param()
    p.Floating_RXFFE = True
    out = OptFom_Create_Output(SimpleNamespace(), _BEST_full(), 0.0, _chdata(),
                               p, _op())
    assert hasattr(out, 'floating_tap_locations')
    assert not hasattr(out, 'floating_tap_coef')


def test_octave_IR_is_zero_padded_to_the_sbr_length_when_not_TDMODE():
    """COM Octave, OP.TDMODE=0: BEST.IR(end+1:length(BEST.sbr))=0."""
    op = _op()
    op.TDMODE = False
    BEST = _BEST_full()
    out = OptFom_Create_Output(SimpleNamespace(), BEST, 0.0, _chdata(),
                               _param(), op)
    assert len(out.IR) == len(BEST.sbr)
    assert np.all(out.IR[100:] == 0)
