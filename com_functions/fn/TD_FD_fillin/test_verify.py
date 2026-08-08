"""Verification tests for TD_FD_fillin().

# ============================================================
# MATLAB GROUND TRUTH (lines 4610-4674)
# Computes frequency-domain fields from time-domain pulse response.
# IL_conv = fd[:f75] / (Vf*M*2) / prr / H_ftr.
# IL fields are set on all chdata channels.
# SDDch, SDDp2p initialised from first channel.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.TD_FD_fillin.py_impl import TD_FD_fillin


def _param(M=8, fb=25e9):
    return SimpleNamespace(
        fb=fb,
        ui=1.0 / fb,
        samples_per_ui=M,
        N_v=10,
        BTorder=2,
        fb_BT_cutoff=1.0,
        fb_BW_cutoff=1.0,
    )


def _op():
    return SimpleNamespace(Bessel_Thomson=False, Butterworth=False)


def _chdata(M=8, fb=25e9, N=200):
    dt = 1.0 / (fb * M)
    T = np.arange(N) * dt
    ir = np.zeros(N)
    ir[0] = 1.0  # unit impulse at t=0
    pulse = np.cumsum(ir)
    cd = SimpleNamespace(
        uneq_pulse_response=pulse,
        t=T,
        type='THRU',
    )
    return [cd]


def test_returns_four_outputs():
    """TD_FD_fillin returns (chdata, param, SDDch, SDDp2p)."""
    result = TD_FD_fillin(_param(), _op(), _chdata())
    assert len(result) == 4


def test_il_fields_set():
    """sdd21_raw field is set on chdata[0]."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _chdata())
    assert hasattr(chdata[0], 'sdd21_raw')


def test_faxis_set():
    """faxis is set on chdata[0]."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _chdata())
    assert hasattr(chdata[0], 'faxis')
    assert chdata[0].faxis[0] == pytest.approx(0.0)


def test_sddch_shape():
    """SDDch has shape (n_freq, 2, 2)."""
    _, _, SDDch, _ = TD_FD_fillin(_param(), _op(), _chdata())
    assert SDDch.ndim == 3
    assert SDDch.shape[1] == 2
    assert SDDch.shape[2] == 2


def test_zero_fields_are_zero():
    """sdd11_raw (zero field) is all zeros."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _chdata())
    assert np.all(chdata[0].sdd11_raw == 0)
