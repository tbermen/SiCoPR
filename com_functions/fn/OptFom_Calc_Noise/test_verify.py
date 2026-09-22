"""Verification tests for OptFom_Calc_Noise().

# ============================================================
# MATLAB GROUND TRUTH (lines 2881-3009)
# Computes sigma_TX, h_J, sigma_ISI, sigma_J, total_noise_rms.
# Non-MMSE, non-RX_CALIBRATION path raises NotImplementedError (get_xtlk_noise).
# RX_CALIBRATION=True: sigma_XT=0, total_noise_rms computable.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Calc_Noise.py_impl import OptFom_Calc_Noise


def _param(M=8, fb=25e9, ndfe=3):
    return SimpleNamespace(
        samples_per_ui=M, fb=fb, ndfe=ndfe,
        levels=4, R_LM=0.4, SNR_TX=30.0,
        sigma_X=1.0, A_DD=0.01, sigma_RJ=0.001,
        cursor_index=1,
        eta_0=0.001,
        sample_dt=1.0 / (M * fb),
    )


def _op_calibration():
    return SimpleNamespace(
        RX_CALIBRATION=True,
        LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=False,
        SNR_TXwC0=False,
        EXE_MODE=0,
        RxFFE=False,
        RxFFE_with_MMSE=False,
    )


def _sbr(cursor_i=40, M=8, N=300):
    sbr = np.zeros(N)
    sbr[cursor_i] = 1.0
    sbr[cursor_i + M] = 0.2
    sbr[cursor_i + 2 * M] = 0.1
    return sbr


def _THIS(cursor_i=40, N=300):
    sbr = _sbr(cursor_i)
    return SimpleNamespace(
        cursor_i=cursor_i,
        A_s=0.5,
        txffe=np.array([0.0, 1.0, 0.0]),
        C=None,
        PSD_results=None,
        sigma_ne=0.005,
        precursors=np.array([0.05, 0.02]),
        far_cursors=np.array([0.01]),
        excess_dfe_cursors=np.array([0.03, 0.01, 0.005]),
        H_ctf=np.ones(512),
        sigma_N=0.02,
    )


def _SETTINGS(N_f=512, fb=25e9):
    f = np.linspace(0, fb/2, N_f)
    return SimpleNamespace(
        phase_memory=np.ones((N_f, 3), dtype=complex),
        H_sy=np.ones(N_f),
        H_r=np.ones(N_f, dtype=complex),
    )


def _chdata(N_f=512, fb=25e9):
    f = np.linspace(0, fb/2, N_f)
    return [SimpleNamespace(faxis=f)]


def test_returns_two_outputs():
    """OptFom_Calc_Noise returns (THIS, abort_status)."""
    sbr = _sbr()
    result = OptFom_Calc_Noise(_THIS(), 0.0, sbr, _SETTINGS(), _chdata(), _param(), _op_calibration())
    assert len(result) == 2


def test_abort_zero_calibration():
    """abort_status=0 for RX_CALIBRATION path (no get_xtlk_noise needed)."""
    sbr = _sbr()
    THIS, abort = OptFom_Calc_Noise(_THIS(), 0.0, sbr, _SETTINGS(), _chdata(), _param(), _op_calibration())
    assert abort == 0


def test_total_noise_rms_positive():
    """total_noise_rms is positive."""
    sbr = _sbr()
    THIS, _ = OptFom_Calc_Noise(_THIS(), 0.0, sbr, _SETTINGS(), _chdata(), _param(), _op_calibration())
    assert THIS.total_noise_rms > 0


def test_h_J_set():
    """h_J is set on THIS."""
    sbr = _sbr()
    THIS, _ = OptFom_Calc_Noise(_THIS(), 0.0, sbr, _SETTINGS(), _chdata(), _param(), _op_calibration())
    assert hasattr(THIS, 'h_J')


def test_sigma_TX_positive():
    """sigma_TX > 0."""
    sbr = _sbr()
    THIS, _ = OptFom_Calc_Noise(_THIS(), 0.0, sbr, _SETTINGS(), _chdata(), _param(), _op_calibration())
    assert THIS.sigma_TX > 0


def test_non_calibration_returns_sigma_xt_zero_without_aggressors():
    """Non-RX_CALIBRATION path: with no aggressor channels, sigma_XT = 0."""
    sbr = _sbr()
    op = _op_calibration()
    op.RX_CALIBRATION = False
    op.RxFFE = False
    # chdata has only one entry (no aggressors) → sigma_XT = 0
    THIS, status = OptFom_Calc_Noise(_THIS(), 0.0, sbr, _SETTINGS(), _chdata(), _param(), op)
    assert hasattr(THIS, 'total_noise_rms')
    assert THIS.total_noise_rms >= 0


# ---------------------------------------------------------------------------
# Added 2026-09-22 to cover an implementation change that arrived without a
# test (an agent was cut off mid-batch). The Octave evidence for the change is
# in py_impl.py; these tests are written from the MATLAB indexing semantics and
# deliberately carry no oracle marker, so this function still counts as needing
# an oracle pass.
#
# MATLAB indexes sbr(cursor_i-1 + M*k) and sbr(cursor_i+1 + M*k) directly for
# k = -1..ndfe, so a span reaching past either end of sbr is an error there.
# Dropping the out-of-range entries, as the port did, silently shortened h_J
# and so changed sigma_J.
# ---------------------------------------------------------------------------

def _op_dfe_span():
    """The guarded branch only runs under LIMIT_JITTER_CONTRIB_TO_DFE_SPAN."""
    op = _op_calibration()
    op.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN = True
    return op


def test_jitter_span_below_the_start_of_sbr_is_refused():
    """cursor_i small enough that cursor_i-1-M lands before sample 1."""
    with pytest.raises(IndexError):
        OptFom_Calc_Noise(_THIS(cursor_i=3), 0.0, _sbr(cursor_i=3),
                          _SETTINGS(), _chdata(), _param(), _op_dfe_span())


def test_jitter_span_past_the_end_of_sbr_is_refused():
    """cursor_i close enough to the end that cursor_i+1+M*ndfe runs off it."""
    # cursor_i must be low enough that _sbr() can still write its 2*M tail
    # (otherwise the helper raises and the test passes for the wrong reason)
    # and high enough that cursor_i+1+M*ndfe runs past the end: 275..283.
    N, ci = 300, 280
    with pytest.raises(IndexError):
        OptFom_Calc_Noise(_THIS(cursor_i=ci, N=N), 0.0, _sbr(cursor_i=ci, N=N),
                          _SETTINGS(), _chdata(), _param(), _op_dfe_span())


def test_jitter_span_well_inside_sbr_still_answers():
    """The guard must not refuse an ordinary cursor position."""
    THIS, abort = OptFom_Calc_Noise(_THIS(), 0.0, _sbr(), _SETTINGS(),
                                    _chdata(), _param(), _op_dfe_span())
    assert abort == 0
    assert len(np.asarray(THIS.h_J)) == 5      # k = -1..ndfe with ndfe=3


# ============================================================
# COM Octave 4p16p0 oracle pins.
# OptFom_Calc_Noise extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave on the fixtures below.
# get_xtlk_noise is replaced, in Octave and here, by a stub returning the
# sentinel 999, so the probe shows WHICH branch the reference takes without
# dragging in the crosstalk machinery.
#
# Common fixture: M=8, fb=25e9, ndfe=3, levels=4, R_LM=0.4, SNR_TX=30,
# sigma_X=1, A_DD=0.01, sigma_RJ=0.001, eta_0=0.001, cursor_index=1,
# RxFFE_cmx=RxFFE_cpx=1; THIS.cursor_i=40 (Octave 41), A_s=0.5,
# txffe=[0 1 0], sigma_ne=0.005, precursors=[0.05 0.02], far=[0.01],
# excess=[0.03 0.01 0.005], H_ctf=ones, sigma_N=0.02; phase_memory=ones,
# H_sy=ones, H_r=ones, faxis=linspace(0,fb/2,512).
# ============================================================
_XT_SENTINEL = 999.0


def _stub_xtlk(monkeypatch):
    """Same sentinel the Octave probe used for get_xtlk_noise."""
    import com_functions.fn.OptFom_Calc_Noise.py_impl as mod
    monkeypatch.setattr(mod, '_get_xtlk_noise',
                        lambda *a, **k: (_XT_SENTINEL, 0, 0))


def _param_ffe(**kw):
    p = _param()
    p.RxFFE_cmx, p.RxFFE_cpx = 1, 1
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _SETTINGS6(N_f=512):
    """phase_memory wide enough for column ii+cmx+1+length(txffe) = 5, 6."""
    s = _SETTINGS(N_f)
    s.phase_memory = np.ones((N_f, 6), dtype=complex)
    return s


def _smooth_sbr(cursor_i=40, M=8, N=300):
    n = np.arange(N)
    return (np.exp(-((n - cursor_i) / 6.0) ** 2)
            + 0.3 * np.exp(-((n - cursor_i - 2 * M) / 9.0) ** 2))


_OCT_SIGMA_TX = 0.11858541225631422
_OCT_TOTAL_IMPULSE = 0.13606064824187777      # h_J identically zero
_OCT_TOTAL_SMOOTH = 0.13627109700825152       # smooth pulse, no span limit
_OCT_TOTAL_SMOOTH_SPAN = 0.13627046567499207  # smooth pulse, DFE-span limit
# h_J for the smooth pulse, first eight of thirty-seven (the tail underflows)
_OCT_HJ_SMOOTH_HEAD = [1.1471780370576782e-11, 1.5527279619307519e-06,
                       0.0064234331020327252, 0.6051206820137176,
                       0.040753818906376527, -0.39003169473026111,
                       -0.0064167267296981745, -0.21387597926418084]
# h_J for the smooth pulse under LIMIT_JITTER_CONTRIB_TO_DFE_SPAN, k=-1..ndfe
_OCT_HJ_SMOOTH_SPAN = [0.6051206820137176, 0.040753818906376527,
                       -0.39003169473026111, -0.0064167267296981745,
                       -0.21387597926418084]


def test_octave_sigma_tx_isi_and_total_noise():
    """COM Octave, RX_CALIBRATION path: sigma_TX, ISI_N, total_noise_rms."""
    THIS, abort = OptFom_Calc_Noise(_THIS(), 0.0, _sbr(), _SETTINGS(),
                                    _chdata(), _param(), _op_calibration())
    assert abort == 0
    assert THIS.sigma_TX == pytest.approx(_OCT_SIGMA_TX, rel=1e-14)
    assert THIS.ISI_N == pytest.approx(0.01, rel=1e-14)
    assert THIS.sigma_N == pytest.approx(0.02, rel=1e-14)
    assert THIS.total_noise_rms == pytest.approx(_OCT_TOTAL_IMPULSE, rel=1e-14)


def test_octave_h_J_sampling_offset_no_span_limit():
    """COM Octave: without the span limit h_J walks sbr(offset-1:M:end)
    against sbr(offset+1:M:end), offset = mod(cursor_i, M) bumped past 1."""
    THIS, _ = OptFom_Calc_Noise(_THIS(), 0.0, _smooth_sbr(), _SETTINGS(),
                                _chdata(), _param(), _op_calibration())
    h_J = np.asarray(THIS.h_J).ravel()
    assert len(h_J) == 37
    np.testing.assert_allclose(h_J[:8], _OCT_HJ_SMOOTH_HEAD, rtol=1e-13)
    assert THIS.total_noise_rms == pytest.approx(_OCT_TOTAL_SMOOTH, rel=1e-14)


def test_octave_h_J_under_dfe_span_limit():
    """COM Octave: the span-limited h_J is sbr(cursor_i-+1 + M*(-1:ndfe))."""
    THIS, _ = OptFom_Calc_Noise(_THIS(), 0.0, _smooth_sbr(), _SETTINGS(),
                                _chdata(), _param(), _op_dfe_span())
    np.testing.assert_allclose(np.asarray(THIS.h_J).ravel(),
                               _OCT_HJ_SMOOTH_SPAN, rtol=1e-13)
    assert THIS.total_noise_rms == pytest.approx(_OCT_TOTAL_SMOOTH_SPAN,
                                                 rel=1e-14)


def test_octave_ffe_opt_method_strcmp_is_case_sensitive(monkeypatch):
    """COM Octave: strcmp(OP.FFE_OPT_METHOD,'MMSE') does no case folding and
    no trimming, so only the exact string reaches PSD_results.S_xn_rms.

    Everything but sigma_XT is zeroed, so total_noise_rms == sigma_XT.
    Octave gave 111 for MMSE and 999 for mmse, Mmse and 'MMSE '.
    An .upper() here sent all four to the PSD branch.
    """
    _stub_xtlk(monkeypatch)
    param = _param_ffe(A_DD=0.0, sigma_RJ=0.0, eta_0=0.0)
    seen = {}
    for method in ('MMSE', 'mmse', 'Mmse', 'MMSE '):
        THIS = _THIS()
        THIS.A_s = 0.0
        THIS.sigma_ne = 0.0
        THIS.precursors = np.zeros(1)
        THIS.far_cursors = np.zeros(1)
        THIS.excess_dfe_cursors = np.zeros(1)
        THIS.sigma_N = 0.0
        THIS.C = np.array([1.0, 0.0, 0.0])
        THIS.PSD_results = SimpleNamespace(S_xn_rms=111.0, S_n_rms=222.0)
        op = _op_calibration()
        op.RX_CALIBRATION = False
        op.RxFFE = True
        op.RxFFE_with_MMSE = False
        op.FFE_OPT_METHOD = method
        out, _ = OptFom_Calc_Noise(THIS, 0.0, _sbr(), _SETTINGS6(), _chdata(),
                                   param, op)
        seen[method] = out.total_noise_rms
    assert seen['MMSE'] == pytest.approx(111.0, rel=1e-14)
    assert seen['mmse'] == pytest.approx(999.0, rel=1e-14)
    assert seen['Mmse'] == pytest.approx(999.0, rel=1e-14)
    assert seen['MMSE '] == pytest.approx(999.0, rel=1e-14)


def test_octave_rxffe_sigma_N_recompute(monkeypatch):
    """COM Octave: with RxFFE and no MMSE, sigma_N is rebuilt from H_Rx_FFE."""
    _stub_xtlk(monkeypatch)
    THIS = _THIS()
    THIS.C = np.array([0.2, 1.0, -0.1])
    THIS.PSD_results = SimpleNamespace(S_xn_rms=111.0, S_n_rms=222.0)
    op = _op_calibration()
    op.RX_CALIBRATION = False
    op.RxFFE = True
    op.FFE_OPT_METHOD = 'mmse'
    out, _ = OptFom_Calc_Noise(THIS, 0.0, _sbr(), _SETTINGS6(), _chdata(),
                               _param_ffe(), op)
    assert out.sigma_N == pytest.approx(0.12298373876248793, rel=1e-13)
    assert out.total_noise_rms == pytest.approx(999.00001663538524, rel=1e-14)


def test_octave_short_rxffe_tap_vector_is_refused(monkeypatch):
    """COM Octave: C(ii+RxFFE_cmx+1) is a direct index, so a C shorter than
    cmx+cpx+1 errors with 'C(3): out of bound 2'.  Skipping the missing taps
    reported a sigma_N the reference will not produce."""
    _stub_xtlk(monkeypatch)
    THIS = _THIS()
    THIS.C = np.array([0.0, 1.0])
    THIS.PSD_results = SimpleNamespace(S_xn_rms=111.0, S_n_rms=222.0)
    op = _op_calibration()
    op.RX_CALIBRATION = False
    op.RxFFE = True
    op.FFE_OPT_METHOD = 'mmse'
    with pytest.raises(IndexError):
        OptFom_Calc_Noise(THIS, 0.0, _sbr(), _SETTINGS6(), _chdata(),
                          _param_ffe(), op)


def test_octave_cursor_index_zero_and_past_the_end():
    """COM Octave, OP.SNR_TXwC0=1: txffe(0) errors with 'subscripts must be
    either integers 1 to (2^63)-1 or logicals', and txffe(4) of three taps
    with 'out of bound 3'.  txffe[cursor_index-1] wrapped a zero cursor_index
    round to the LAST tap instead of refusing."""
    op = _op_calibration()
    op.SNR_TXwC0 = True
    p0 = _param()
    p0.cursor_index = 0
    with pytest.raises(IndexError):
        OptFom_Calc_Noise(_THIS(), 0.0, _sbr(), _SETTINGS(), _chdata(), p0, op)
    p4 = _param()
    p4.cursor_index = 4          # txffe has three taps
    with pytest.raises(IndexError):
        OptFom_Calc_Noise(_THIS(), 0.0, _sbr(), _SETTINGS(), _chdata(), p4, op)


def test_octave_rxffe_with_mmse_total_is_norm_of_three():
    """COM Octave, RxFFE_with_MMSE: total is norm([sigma_ISI S_n_rms sigma_ne]).

    S_n_rms is 0.05 here, comparable to sigma_ISI (0.0632) and sigma_ne
    (0.005), so every one of the three terms moves the answer; the crosstalk
    and jitter terms of the other branch must not appear.
    """
    THIS = _THIS()
    THIS.PSD_results = SimpleNamespace(S_xn_rms=111.0, S_n_rms=0.05)
    op = _op_calibration()
    op.RX_CALIBRATION = False
    op.RxFFE_with_MMSE = True
    out, _ = OptFom_Calc_Noise(THIS, 0.0, _sbr(), _SETTINGS(), _chdata(),
                               _param(), op)
    assert out.total_noise_rms == pytest.approx(0.08093207028119323, rel=1e-14)
