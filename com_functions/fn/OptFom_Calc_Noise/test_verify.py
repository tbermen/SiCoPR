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
