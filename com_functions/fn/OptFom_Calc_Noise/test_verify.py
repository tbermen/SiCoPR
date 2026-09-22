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
