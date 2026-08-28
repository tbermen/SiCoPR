"""Verification tests for RILN_TD().

# ============================================================
# MATLAB GROUND TRUTH (lines 4226-4305)
# Computes ILN = FIT.PR - REF.PR over a range from ipeak.
# All-zero sdd21/RIL: uses eps-valued impulse response (s21_to_impulse_DC zero path).
# Non-zero sdd21: raises NotImplementedError (interp_Sparam pending).
# Returns struct with REF, FIT, ILN, FOM, FOM_PDF, SNR_ISI_FOM.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sicopr  # assembled module (provides s21_to_impulse_DC, Bessel/Butterworth filters)
import com_functions.fn.RILN_TD.py_impl as _mod
from com_functions.fn.RILN_TD.py_impl import RILN_TD

# Inject the real top-level dependencies RILN_TD calls as bare globals.
for _n in dir(sicopr):
    _v = getattr(sicopr, _n)
    if callable(_v) and not _n.startswith('__') and not hasattr(_mod, _n):
        setattr(_mod, _n, _v)


def _op():
    return SimpleNamespace(
        transmitter_transition_time=10e-12,
        EC_PULSE_TOL=0.01,
        EC_REL_TOL=1e-4,
        EC_DIFF_TOL=1e-6,
        ENFORCE_CAUSALITY=0,
        impulse_response_truncation_threshold=1e-3,
        BinSize=1e-4,
        DEBUG=True,  # downgrade anti-causal check to a warning for synthetic channels
    )


def _param(M=8, fb=25e9):
    return SimpleNamespace(
        fb=fb, ui=1.0/fb, samples_per_ui=M,
        fb_BW_cutoff=1.0, BTorder=2, fb_BT_cutoff=1.0,
        sample_dt=1.0/(2*fb),
        levels=4,
        specBER=1e-6,
    )


def _freq(N=100, fmax=50e9):
    return np.linspace(0, fmax, N)


def test_returns_struct():
    """RILN_TD returns a SimpleNamespace with REF, FIT, ILN."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    RIL = np.zeros(len(f))
    result = RILN_TD(sdd21, RIL, f, _op(), _param())
    assert hasattr(result, 'REF')
    assert hasattr(result, 'FIT')
    assert hasattr(result, 'ILN')


def test_ILN_zero_for_identical_channels():
    """ILN ≈ 0 when sdd21 == RIL (both zero)."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    result = RILN_TD(sdd21, sdd21.copy(), f, _op(), _param())
    assert np.allclose(result.ILN, 0.0, atol=1e-10)


def test_FOM_nonnegative():
    """FOM is non-negative."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    result = RILN_TD(sdd21, sdd21.copy(), f, _op(), _param())
    assert result.FOM >= 0


def test_REF_PR_length_positive():
    """REF.PR has positive length."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    result = RILN_TD(sdd21, sdd21.copy(), f, _op(), _param())
    assert len(result.REF.PR) > 0


def _lossy(f, tau=150e-12, loss_db_per_ghz=0.6, scale=1.0):
    mag = scale * np.exp(-(loss_db_per_ghz / 20.0) * np.log(10) * f / 1e9)
    return mag * np.exp(-1j * 2 * np.pi * f * tau)


def test_nonzero_channel_computes_riln():
    """Realistic lossy sdd21/RIL: RILN_TD runs end-to-end (s21_to_impulse_DC is
    implemented now) and returns finite REF/FIT/ILN/FOM (no NotImplementedError)."""
    f = _freq()
    sdd21 = _lossy(f, tau=150e-12)
    RIL = _lossy(f, tau=300e-12, scale=0.2)  # weaker reflective path
    result = RILN_TD(sdd21, RIL, f, _op(), _param())
    assert hasattr(result, 'REF') and hasattr(result, 'FIT') and hasattr(result, 'ILN')
    assert len(result.REF.PR) > 0
    assert np.isfinite(float(result.FOM))


def test_uses_real_s21_to_impulse_DC():
    assert _mod.s21_to_impulse_DC is sicopr.s21_to_impulse_DC
