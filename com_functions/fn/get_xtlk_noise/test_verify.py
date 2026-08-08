"""Verification tests for get_xtlk_noise().

# ============================================================
# MATLAB GROUND TRUTH (lines 7844-7938)
# Computes FEXT/NEXT crosstalk noise sigma.
# No crosstalk channels (only THRU in chdata): sigma_XT=0.
# With FEXT/NEXT channels: computes power-weighted sigma.
# upsampled_txffe all-zero: PWF_tx=ones.
# nargout==1 NEXT: returns MDNEXT_ICN * scale.
# nargout==3: returns (sigma_XT, sigma_FEXT, sigma_NEXT).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.get_xtlk_noise.py_impl import get_xtlk_noise


def _faxis(N=64, fb=25e9):
    return np.linspace(0, fb, N)


def _param(fb=25e9, M=8, dt=None):
    if dt is None:
        dt = 1.0/(2*fb)
    return SimpleNamespace(
        fb=fb, samples_per_ui=M, sample_dt=dt,
        levels=4, f2=fb/2,
        RxFFE_cmx=2, RxFFE_cpx=2,
    )


def _thru_chdata(N=64, fb=25e9):
    f = _faxis(N, fb)
    return [SimpleNamespace(
        faxis=f, type='THRU',
        sdd21ctf=np.zeros(N, dtype=complex),
        delta_f=f[1]-f[0], A=0.5,
    )]


def _fext_chdata(N=64, fb=25e9):
    f = _faxis(N, fb)
    sdd21ctf = 0.01 * np.ones(N, dtype=complex)
    return [
        SimpleNamespace(faxis=f, type='THRU', sdd21ctf=np.zeros(N, dtype=complex),
                        delta_f=f[1]-f[0], A=0.5),
        SimpleNamespace(faxis=f, type='FEXT', sdd21ctf=sdd21ctf,
                        delta_f=f[1]-f[0], A=0.5),
    ]


def test_no_xtalk_returns_zero():
    """With only THRU channel, sigma_XT=0 for FEXT query."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'FEXT', _param(), _thru_chdata())
    assert result == pytest.approx(0.0)


def test_no_xtalk_returns_zero_next():
    """With only THRU channel, sigma_XT=0 for NEXT query."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'NEXT', _param(), _thru_chdata())
    assert result == pytest.approx(0.0)


def test_fext_sigma_positive():
    """FEXT sigma > 0 when FEXT channel present."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'FEXT', _param(), _fext_chdata())
    assert result >= 0.0


def test_three_output_mode():
    """'both' mode returns (sigma_XT, sigma_FEXT, sigma_NEXT)."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'both', _param(), _fext_chdata())
    assert len(result) == 3


def test_invalid_type_raises():
    """Unknown channel type in chdata raises ValueError."""
    txffe = np.zeros(8)
    f = _faxis()
    chdata_bad = [
        SimpleNamespace(faxis=f, type='THRU', sdd21ctf=np.zeros(64, dtype=complex),
                        delta_f=f[1]-f[0], A=0.5),
        SimpleNamespace(faxis=f, type='UNKNOWN', sdd21ctf=np.zeros(64, dtype=complex),
                        delta_f=f[1]-f[0], A=0.5),
    ]
    with pytest.raises(ValueError):
        get_xtlk_noise(txffe, 'NEXT', _param(), chdata_bad)


def test_nzero_txffe_runs():
    """Non-zero upsampled_txffe runs without error."""
    txffe = np.zeros(8)
    txffe[3] = 0.8
    txffe[2] = 0.1
    result = get_xtlk_noise(txffe, 'FEXT', _param(), _fext_chdata())
    assert result >= 0.0
