"""Integration tests for get_ILN_cmp_td (MATLAB lines ~6270).

Computes the time-domain insertion-loss noise (TD ILN): fits a smooth complex
model to sdd21, builds REF (raw) and FIT (fitted) pulse responses through the
TX/Bessel-Thomson filters, and reports a FOM from their difference PDF.

Bessel_Thomson_Filter / Butterworth_Filter / s21_to_impulse_DC are top-level fns
in the assembled com.py, so this is an INTEGRATION test: the real dependencies
are injected from `com`. Both the zero-input (degenerate) and a realistic lossy
channel are exercised.
"""
import os
import sys
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com
import com_functions.fn.get_ILN_cmp_td.py_impl as _mod
from com_functions.fn.get_ILN_cmp_td.py_impl import get_ILN_cmp_td

for _n in dir(com):
    _v = getattr(com, _n)
    if callable(_v) and not _n.startswith('__') and not hasattr(_mod, _n):
        setattr(_mod, _n, _v)


def _op():
    return SimpleNamespace(
        BinSize=1e-4,
        transmitter_transition_time=10e-12,
        impulse_response_truncation_threshold=1e-7,
        DEBUG=True,  # downgrade the anti-causal check to a warning for synthetic channels
        ENFORCE_CAUSALITY=0,
        EC_PULSE_TOL=0.01, EC_REL_TOL=1e-2, EC_DIFF_TOL=1e-3,
    )


def _param(fb=26.5625e9, spui=4):
    p = SimpleNamespace()
    p.samples_per_ui = spui
    p.samples_for_C2M = 8
    p.sample_dt = 1.0 / (fb * spui)
    p.specBER = 1e-4
    p.levels = 4
    p.fb = fb
    p.BTorder = 4
    p.fb_BT_cutoff = 0.75
    p.fb_BW_cutoff = 0.75
    return p


def _channel(fb=26.5625e9, n=200, tau=200e-12, loss_db_per_ghz=0.6):
    f = np.linspace(0, 2 * fb, n)
    mag = np.exp(-(loss_db_per_ghz / 20.0) * np.log(10) * f / 1e9)
    sdd21 = mag * np.exp(-1j * 2 * np.pi * f * tau)
    return sdd21, f


_TD_FIELDS = ('FOM', 'FOM_PDF', 'SNR_ISI_FOM', 'SNR_ISI_FOM_PDF', 'REF', 'FIT')


def test_zero_sdd21_returns_iln():
    N = 50
    f = np.linspace(0, 26.5625e9, N)
    sdd21 = np.zeros(N, dtype=complex)
    ILN, efit, TD = get_ILN_cmp_td(sdd21, f, _op(), _param())
    assert len(ILN) == N
    assert len(efit) == N


def test_zero_sdd21_td_struct_fields():
    N = 50
    f = np.linspace(0, 26.5625e9, N)
    ILN, efit, TD = get_ILN_cmp_td(np.zeros(N, dtype=complex), f, _op(), _param())
    for field in _TD_FIELDS:
        assert hasattr(TD, field), f'missing TD field: {field}'


def test_nonzero_channel_computes_iln():
    """Realistic lossy channel: ILN/efit have input length and TD has all fields."""
    sdd21, f = _channel()
    ILN, efit, TD = get_ILN_cmp_td(sdd21, f, _op(), _param())
    assert len(ILN) == len(f)
    assert len(efit) == len(f)
    assert np.all(np.isfinite(efit))
    for field in _TD_FIELDS:
        assert hasattr(TD, field), f'missing TD field: {field}'
    assert np.isfinite(float(TD.FOM))


def test_iln_shape_matches_input():
    N = 30
    f = np.linspace(0, 26.5625e9, N)
    ILN, efit, TD = get_ILN_cmp_td(np.zeros(N, dtype=complex), f, _op(), _param())
    assert len(ILN) == N
    assert len(efit) == N


def test_uses_real_s21_to_impulse_DC():
    assert _mod.s21_to_impulse_DC is com.s21_to_impulse_DC
    assert _mod.Bessel_Thomson_Filter is com.Bessel_Thomson_Filter


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
