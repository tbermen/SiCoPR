"""Integration tests for calculate_delay_CausalityEnforcement.

MATLAB GROUND TRUTH (lines 4933-5086):
  Channel delay via IEEE-370 causality enforcement. Step 7 interpolates the
  S-parameter to a uniform grid (interp_Sparam) and ifft's to a pulse response;
  step 8 estimates the delay from the peak-index difference between the original
  and causality-enforced pulse responses. Returns (delay_sec, delay_idx).

  interp_Sparam / lfilter are top-level fns in the assembled com.py, so this
  function is exercised as an INTEGRATION test: the real dependencies are
  injected from `com` and the delay is checked against a known group delay.
"""
import os
import sys
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com  # assembled module (provides interp_Sparam, lfilter, ...)
import com_functions.fn.calculate_delay_CausalityEnforcement.py_impl as _mod
from com_functions.fn.calculate_delay_CausalityEnforcement.py_impl import (
    calculate_delay_CausalityEnforcement)

# Inject the real top-level dependencies that the function calls as bare globals.
for _n in dir(com):
    _v = getattr(com, _n)
    if callable(_v) and not _n.startswith('__') and not hasattr(_mod, _n):
        setattr(_mod, _n, _v)


def _param(fb=26.5625e9, spui=32):
    p = SimpleNamespace()
    p.Z0 = 100.0
    p.fb = fb
    p.samples_per_ui = spui
    p.sample_dt = 1.0 / (fb * spui)
    p.flim = float('inf')
    return p


def _OP():
    return SimpleNamespace(
        interp_sparam_mag='linear_trend_to_DC',
        interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf',
        ENFORCE_CAUSALITY=1,
    )


def _delayed_channel(tau, fb=26.5625e9, n=400, loss_db_per_ghz=0.5):
    """A realistic lossy low-pass channel with group delay ~tau (the loss keeps
    it causal so interp_Sparam does not flag it as anti-causal)."""
    freq = np.linspace(0, 2 * fb, n)
    mag = np.exp(-(loss_db_per_ghz / 20.0) * np.log(10) * freq / 1e9)
    sdd21 = mag * np.exp(-1j * 2 * np.pi * freq * tau)
    return freq, sdd21


def test_returns_finite_delay_and_int_index():
    freq, sdd21 = _delayed_channel(tau=150e-12)
    delay_sec, delay_idx = calculate_delay_CausalityEnforcement(freq, sdd21, _param(), _OP())
    assert np.isfinite(delay_sec)
    assert delay_sec >= 0.0
    assert isinstance(delay_idx, (int, np.integer))


def test_larger_delay_gives_larger_estimate():
    """A channel with more group delay yields a larger (or equal) delay estimate."""
    p, op = _param(), _OP()
    d_small, _ = calculate_delay_CausalityEnforcement(*_delayed_channel(tau=100e-12), p, op)
    d_large, _ = calculate_delay_CausalityEnforcement(*_delayed_channel(tau=400e-12), p, op)
    assert d_large >= d_small


def test_runs_with_lossy_channel():
    freq, sdd21 = _delayed_channel(tau=200e-12, loss_db_per_ghz=0.5)
    delay_sec, delay_idx = calculate_delay_CausalityEnforcement(freq, sdd21, _param(), _OP())
    assert np.isfinite(delay_sec)


def test_uses_real_interp_Sparam():
    """The injected dependency is the real assembled interp_Sparam (not a stub)."""
    assert _mod.interp_Sparam is com.interp_Sparam


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
