"""Integration tests for calculate_delay_CausalityEnforcement.

MATLAB GROUND TRUTH (lines 4933-5086):
  Channel delay via IEEE-370 causality enforcement. Step 7 interpolates the
  S-parameter to a uniform grid (interp_Sparam) and ifft's to a pulse response;
  step 8 estimates the delay from the peak-index difference between the original
  and causality-enforced pulse responses. Returns (delay_sec, delay_idx).

  interp_Sparam / lfilter are top-level fns in the assembled sicopr.py, so this
  function is exercised as an INTEGRATION test: the real dependencies are
  injected from `sicopr` and the delay is checked against a known group delay.
"""
import os
import sys
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sicopr  # assembled module (provides interp_Sparam, lfilter, ...)
import com_functions.fn.calculate_delay_CausalityEnforcement.py_impl as _mod
from com_functions.fn.calculate_delay_CausalityEnforcement.py_impl import (
    calculate_delay_CausalityEnforcement)

# Inject the real top-level dependencies that the function calls as bare globals.
for _n in dir(sicopr):
    _v = getattr(sicopr, _n)
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
    assert _mod.interp_Sparam is sicopr.interp_Sparam


# ============================================================
# COM Octave oracle values — tools/octave_oracle.py runs
# calculate_delay_CausalityEnforcement (plus interp_Sparam) verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m.  Literals pinned 2026-09-22.
#
# Three divergences these pin:
#  * search_bounds.  MATLAB's min(peak_x_idx, peak_y_idx) is over 1-BASED peak
#    indices, so its shift search is one wider at each end than the 0-based
#    min the port used.  Since the error curve rises monotonically across the
#    searched range, the answer is always the lower bound and the port came out
#    one index high.  On 36 asymmetric channels the pre-fix port matched Octave
#    0/36; the fixed port matches 23/23 (the other 13 are inputs on which
#    interp_Sparam raises 'Anti-causal response found' in both languages).
#  * fout step.  MATLAB's 1/round(fmax/freq_step)*fmax takes the reciprocal
#    FIRST; fmax/nstep is a different rounding and moved 2286 of 3193 grid
#    points by one ulp.
#  * fout end.  MATLAB's colon clamps its final element to the limit; the port
#    let np.arange overshoot fmax.
#
# Tolerance note: the tests below use channels whose pulse response is NOT
# symmetric about a half sample.  A symmetric one makes max() a coin flip on
# ~3e-15 of FFT noise (numpy pocketfft vs Octave FFTW), which says nothing
# about the port.
# ============================================================

def _asym_channel(tau, fb, n, span, a_sqrt, ripple):
    """Skin-effect loss plus a small in-band ripple, so the pulse response has
    no tied peak."""
    freq = np.linspace(0, span * fb, n)
    fg = freq / 1e9
    mag = np.exp(-a_sqrt * np.sqrt(fg) - 0.02 * fg)
    s = mag * np.exp(-1j * 2 * np.pi * freq * tau)
    return freq, s * (1.0 + ripple * np.exp(-1j * 2 * np.pi * freq * 3.1 * tau))


@pytest.mark.parametrize('fb,spui,n,span,tau,a,r,exp_sec,exp_idx', [
    (26.5625e9, 32, 200, 1.0, 120e-12, 0.35, 0.06, 5.7647058823529409e-11, 50),
    (26.5625e9, 32, 400, 2.0, 300e-12, 0.35, 0.06, 2.4117647058823531e-10, 206),
    (26.5625e9, 32, 400, 2.0, 640e-12, 0.80, 0.12, 5.5294117647058828e-10, 471),
    (53.125e9, 64, 301, 1.5, 300e-12, 0.80, 0.12, 2.2441176470588234e-10, 764),
    (53.125e9, 64, 200, 1.0, 640e-12, 0.35, 0.06, 6.1176470588235298e-10, 2081),
])
def test_octave_delay_asymmetric_channels(fb, spui, n, span, tau, a, r,
                                          exp_sec, exp_idx):
    """Pre-fix the port returned exp_idx+1 on every one of these."""
    freq, sdd21 = _asym_channel(tau, fb, n, span, a, r)
    p = _param(fb=fb, spui=spui)
    delay_sec, delay_idx = calculate_delay_CausalityEnforcement(
        freq, sdd21, p, _OP())
    assert delay_idx == exp_idx
    assert delay_sec == exp_sec


def test_octave_fout_grid(monkeypatch):
    """The uniform grid handed to interp_Sparam, captured in flight.

    COM Octave: 0:1/round(fmax/freq_step)*fmax:fmax
      fmax=425000000000, freq_step=133145363.40852131  -> 3193 points
      fmax=1700000000000, freq_step=123377732.24043716 -> 13780 points
    """
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(ILin, fin, fout, *a, **k):
        seen['fout'] = np.asarray(fout).ravel().copy()
        raise _Stop

    monkeypatch.setattr(_mod, 'interp_Sparam', _spy)

    # (a) step form: 1/nstep first, then *fmax
    freq, sdd21 = _asym_channel(300e-12, 26.5625e9, 400, 2.0, 0.35, 0.06)
    with pytest.raises(_Stop):
        calculate_delay_CausalityEnforcement(freq, sdd21, _param(26.5625e9, 32), _OP())
    f = seen['fout']
    assert f.size == 3193
    assert f[0] == 0.0
    assert f[1] == 133145363.40852129
    assert f[2] == 266290726.81704259
    assert f[1000] == 133145363408.5213
    assert f[3191] == 424866854636.59143
    assert f[3192] == 425000000000.0

    # (b) final element clamped to fmax, not allowed to overshoot
    freq, sdd21 = _asym_channel(300e-12, 53.125e9, 733, 1.7, 0.35, 0.06)
    with pytest.raises(_Stop):
        calculate_delay_CausalityEnforcement(freq, sdd21, _param(53.125e9, 64), _OP())
    f = seen['fout']
    assert f.size == 13780
    assert f[1] == 123376152.11553815
    assert f[13778] == 1699876623847.8845
    assert f[13779] == 1700000000000.0, "colon clamps the last point to fmax"


def test_octave_nan_at_first_interp_point_raises(monkeypatch):
    """MATLAB's IL(in)=IL(in-1) reads IL(0) and errors when point 1 is NaN.

    COM Octave: IL=[NaN;1;2;3]; IL_nan=find(isnan(IL));
                for in=IL_nan.'; IL(in)=IL(in-1); end
      -> error: IL(0): subscripts must be either integers 1 to (2^63)-1 or logicals
    """
    def _nan_first(ILin, fin, fout, *a, **k):
        out = np.ones(len(np.asarray(fout).ravel()), dtype=complex)
        out[0] = np.nan
        return out

    monkeypatch.setattr(_mod, 'interp_Sparam', _nan_first)
    freq, sdd21 = _asym_channel(300e-12, 26.5625e9, 200, 1.0, 0.35, 0.06)
    with pytest.raises(IndexError):
        calculate_delay_CausalityEnforcement(freq, sdd21, _param(), _OP())


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
