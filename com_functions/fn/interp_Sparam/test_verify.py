"""Verification tests for interp_Sparam().

# ============================================================
# MATLAB GROUND TRUTH (lines 7950-8165)
# Interpolates S-parameters Sin(fin) → Sout(fout).
# H_mag = abs(Sin), H_ph = unwrap(angle(Sin)).
# Anti-causal check: mean(diff(H_ph)) > 0 → warn/error.
# Magnitude methods: linear_trend_to_DC, old, extrap_to_DC, trend_to_DC, extrap_to_DC_or_zero.
# Phase methods: interp_to_DC, old, zero_DC, interp_and_shift_to_DC, trend_and_shift_to_DC.
# ZERO_PAD=True: applies Tukey window, zeros above fin[-1].
# Returns Sout = H_mag_i * exp(1j*H_ph_i), shape = len(fout).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.interp_Sparam.py_impl import interp_Sparam


def _op(debug=True):
    return SimpleNamespace(DEBUG=debug, ZERO_PAD=False)


def _param():
    p = SimpleNamespace()
    p.fb = 28e9
    p.f_r = 0.75
    return p


def _causal_sin(fin, delay=100e-12):
    """Causal (negative-phase-slope) complex signal for a transmission line."""
    return np.exp(-1j * 2 * np.pi * fin * delay)


def test_B02_D4_mag_floor_uses_machine_eps():
    """MATLAB line 8091: H_mag(H_mag<eps)=eps with eps = machine epsilon
    (2.2204e-16). Pre-fix Python floors at np.finfo(float).tiny (2.2e-308), so a
    magnitude of 1e-18 (below machine eps, above tiny) is NOT floored. After the
    fix, |Sout| is floored to ~2.2e-16.
    """
    fin = np.linspace(1e9, 20e9, 40)
    Sin = 1e-18 * _causal_sin(fin, delay=100e-12)     # |Sin| = 1e-18 everywhere
    Sout = interp_Sparam(Sin, fin, fin, 'old', 'old', _op(), _param())
    assert np.min(np.abs(Sout)) >= 0.9 * np.finfo(float).eps, (
        "min|Sout|=%g; MATLAB floors |S|<eps to machine eps 2.2204e-16 (line 8091). "
        "Pre-fix tiny-floor leaves 1e-18." % np.min(np.abs(Sout)))


def test_B02_D5_phase_extrapolates_linearly():
    """MATLAB line 8185: H_ph_i = interp1(fin,H_ph,fout,'linear','extrap') extends
    the phase slope beyond fin[-1]. Pre-fix Python clamps (left/right), freezing
    the group delay to ~0 in the extrapolated band. After the fix the group delay
    continues at the channel delay.
    """
    delay = 100e-12
    fin = np.linspace(1e9, 20e9, 40)
    fout = np.linspace(1e9, 40e9, 80)                 # extends to 2x fin[-1]
    Sin = _causal_sin(fin, delay)
    Sout = interp_Sparam(Sin, fin, fout, 'old', 'old', _op(), _param())
    ph = np.unwrap(np.angle(Sout))
    gd = -np.diff(ph) / np.diff(fout)                 # group delay
    extrap = fout[1:] > fin[-1]
    assert np.mean(gd[extrap]) > 0.5 * delay, (
        "mean group delay in the extrapolated band = %g s; expected ~%g s "
        "(linear phase extrapolation). Pre-fix clamp gives ~0." %
        (np.mean(gd[extrap]), delay))


def test_output_length_matches_fout():
    """Output has same length as fout."""
    fin = np.linspace(1e9, 20e9, 50)
    fout = np.linspace(0, 25e9, 100)
    Sin = _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'old', 'old', _op(), _param())
    assert len(Sout) == len(fout)


def test_identity_at_input_frequencies():
    """At fin points that coincide with fout, magnitude should match."""
    fin = np.linspace(1e9, 20e9, 20)
    fout = fin.copy()
    Sin = 0.8 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'old', 'old', _op(), _param())
    assert np.allclose(np.abs(Sout), np.abs(Sin), atol=1e-6)


def test_extrap_to_dc_at_zero_freq():
    """extrap_to_DC: first output at fout=0 should be finite."""
    fin = np.linspace(1e9, 20e9, 30)
    fout = np.concatenate([[0.0], fin])
    Sin = 0.7 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'extrap_to_DC', 'interp_to_DC', _op(), _param())
    assert np.isfinite(Sout[0])
    assert len(Sout) == len(fout)


def test_linear_trend_to_dc_dc_value_finite():
    """linear_trend_to_DC: DC extrapolation should be finite and positive magnitude."""
    fin = np.linspace(1e9, 20e9, 30)
    fout = np.concatenate([[0.0], fin])
    Sin = 0.7 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC', 'interp_to_DC', _op(), _param())
    assert np.isfinite(np.abs(Sout[0]))
    assert np.abs(Sout[0]) > 0


def test_anti_causal_raises_in_non_debug():
    """Non-causal response (increasing phase) raises ValueError in non-DEBUG mode."""
    fin = np.linspace(1e9, 20e9, 30)
    # positive phase slope = anti-causal
    Sin = np.exp(1j * 2 * np.pi * fin * 100e-12)
    op = _op(debug=False)
    with pytest.raises(ValueError, match='Anti-causal'):
        interp_Sparam(Sin, fin, fin, 'old', 'old', op, _param())


def test_invalid_mag_method_raises():
    """Invalid opt_interp_Sparam_mag raises ValueError."""
    fin = np.linspace(1e9, 20e9, 20)
    Sin = _causal_sin(fin)
    with pytest.raises(ValueError, match='interp_Sparam: invalid'):
        interp_Sparam(Sin, fin, fin, 'bad_method', 'old', _op(), _param())


def test_trend_to_dc_method():
    """trend_to_DC: returns array of correct length with finite values."""
    fin = np.linspace(1e9, 20e9, 25)
    fout = np.linspace(0, 20e9, 50)
    Sin = 0.6 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'trend_to_DC', 'interp_to_DC', _op(), _param())
    assert len(Sout) == 50
    assert np.all(np.isfinite(Sout))
