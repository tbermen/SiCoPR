"""Tests for capture_RIL_RILN.

MATLAB GROUND TRUTH:
  For a realistic 2-port (small S11/S22, dominant S21):
    rho_port2 from quadratic with Re(Z)>0 selection
    RIL = reflectionless insertion loss (close to S21 for small reflections)
    RILN = RIL - S21

  Quadratic: a*rho^2 + b*rho + c = 0
    a = -S22 + S11*S22*conj(S11) - S21*S12*conj(S11)
  NOTE: S11=S22=0 exactly gives a=0 (degenerate case) → MATLAB errors.
  Always use physically plausible S-params with small but non-zero S11/S22.
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.capture_RIL_RILN.py_impl import capture_RIL_RILN


def _make_chdata(freq, s11, s12, s21, s22):
    ch = SimpleNamespace()
    ch.faxis = np.asarray(freq, dtype=float)
    ch.sdd11_orig = np.asarray(s11, dtype=complex)
    ch.sdd12_orig = np.asarray(s12, dtype=complex)
    ch.sdd21_orig = np.asarray(s21, dtype=complex)
    ch.sdd22_orig = np.asarray(s22, dtype=complex)
    return [ch]


def _typical_chdata(nf=5):
    """Typical lossy 2-port: small reflections, moderate loss."""
    freq = np.linspace(1e9, 10e9, nf)
    phase = -2 * np.pi * freq * 100e-12
    s21 = 0.8 * np.exp(1j * phase)
    s11 = 0.05 * np.ones(nf, dtype=complex)
    s22 = 0.04 * np.ones(nf, dtype=complex)
    return _make_chdata(freq, s11, s21.copy(), s21.copy(), s22)


def test_runs_without_error():
    """Function completes for a typical 2-port channel."""
    rs = capture_RIL_RILN(_typical_chdata())
    assert rs is not None


def test_riln_definition():
    """RILN = RIL - S21 at every frequency."""
    freq = np.array([1e9, 3e9, 10e9])
    phase = -2 * np.pi * freq * 80e-12
    s21 = 0.7 * np.exp(1j * phase)
    s11 = 0.06 + 0.01j * np.ones(3)
    chdata = _make_chdata(freq, s11, s21.copy(), s21.copy(), s11.copy())
    rs = capture_RIL_RILN(chdata)
    assert np.allclose(rs.RILN, rs.RIL - s21, atol=1e-8)


def test_ril_dB_finite():
    """RIL_dB and RILN_dB should be finite."""
    rs = capture_RIL_RILN(_typical_chdata(8))
    assert np.all(np.isfinite(rs.RIL_dB))
    assert np.all(np.isfinite(rs.RILN_dB))


def test_rho_port2_passive():
    """For passive network, |rho_port2| <= 1."""
    rs = capture_RIL_RILN(_typical_chdata(4))
    assert np.all(np.abs(rs.rho_port2) <= 1.0 + 1e-6)


def test_skips_dc_when_faxis_starts_at_zero():
    """When faxis[0]=0, result freq starts at faxis[1]."""
    freq = np.array([0.0, 1e9, 5e9])
    s21 = np.array([0.9 + 0j, 0.85 + 0j, 0.7 + 0j])
    s11 = np.array([0.05 + 0j, 0.06 + 0j, 0.08 + 0j])
    chdata = _make_chdata(freq, s11, s21.copy(), s21.copy(), s11.copy())
    rs = capture_RIL_RILN(chdata)
    assert len(rs.freq) == 2
    assert rs.freq[0] == pytest.approx(1e9)


def test_output_fields_present():
    """Return struct has all required fields."""
    rs = capture_RIL_RILN(_typical_chdata(3))
    for field in ('RIL', 'RIL_dB', 'RILN', 'RILN_dB', 'Z_port1', 'Z_port2',
                  'rho_port1', 'rho_port2', 'freq'):
        assert hasattr(rs, field), f'Missing field: {field}'


def test_freq_length_matches_sdd21():
    """rs.freq has same length as input freq (or freq-1 if DC skipped)."""
    freq = np.array([1e9, 5e9, 10e9, 20e9])
    s21 = 0.6 * np.exp(-1j * 2 * np.pi * freq * 50e-12)
    s11 = 0.07 * np.ones(4, dtype=complex)
    chdata = _make_chdata(freq, s11, s21.copy(), s21.copy(), s11.copy())
    rs = capture_RIL_RILN(chdata)
    assert len(rs.freq) == 4
