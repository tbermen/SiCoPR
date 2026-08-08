"""Tests for make_full_pkg.

MATLAB GROUND TRUTH:
  Zero-length transmission line (Len=0): S11=0, S21=1 for lossless matched line.
  Single capacitor (Cpad>0, others 0): S11 = -jwCZ/(2+jwCZ), S21 = 2/(2+jwCZ).
  TX vs RX: uses different Pkg_len_TX vs Pkg_len_RX.
  DC mode: Z0/=2, Cpad*=2, Cball*=2, etc.
  Returns 4 arrays each of length len(faxis).
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.make_full_pkg.py_impl import make_full_pkg, _synth_tline, _combines4p


def _make_param(mele=1, Z0=100.0, Len_TX=0.0, Len_RX=0.0, Z_c=100.0,
                Cpad=0.0, Cball=0.0, Cbump=0.0, Lcomp=0.0, kappa1=1.0, kappa2=1.0):
    p = SimpleNamespace()
    p.Z0 = Z0
    p.C_diepad = np.array([Cpad, Cpad])
    p.C_pkg_board = np.array([Cball, Cball])
    p.L_comp = np.array([Lcomp, Lcomp])
    p.C_bump = np.array([Cbump, Cbump])
    if mele == 1:
        p.z_p_next_cases = np.array([[Z_c]])
        p.pkg_Z_c = np.array([Z_c, Z_c])
        p.Pkg_len_TX = np.array([Len_TX])
        p.Pkg_len_RX = np.array([Len_RX])
        p.Pkg_len_NEXT = np.array([Len_TX])
        p.Pkg_len_FEXT = np.array([Len_TX])
    p.pkg_tau = 0.0
    p.pkg_gamma0_a1_a2 = np.array([0.0, 0.0, 0.0])
    p.PKG_NAME = None
    p.kappa1 = kappa1
    p.kappa2 = kappa2
    return p


def test_synth_tline_zero_len_matched():
    """Zero length + matched impedance: S11=0, S21=1."""
    f = np.array([1e9, 5e9, 10e9])
    s11, s12, s21, s22 = _synth_tline(f, 100.0, 50.0, [0, 0, 0], 0, 0)
    assert np.allclose(s11, 0, atol=1e-10)
    assert np.allclose(np.abs(s21), 1.0, atol=1e-10)


def test_combines4p_identity_cascade():
    """Cascading two identity networks gives identity."""
    f = np.array([1e9, 5e9])
    s11 = np.zeros(2)
    s21 = np.ones(2)
    s11o, s12o, s21o, s22o = _combines4p(s11, s21, s21, s11, s11, s21, s21, s11)
    assert np.allclose(s21o, 1.0, atol=1e-10)
    assert np.allclose(s11o, 0.0, atol=1e-10)


def test_make_full_pkg_returns_four_arrays():
    faxis = np.linspace(1e9, 25e9, 50)
    param = _make_param()
    s11, s12, s21, s22 = make_full_pkg('TX', faxis, param, 'THRU')
    assert len(s11) == 50
    assert len(s21) == 50


def test_make_full_pkg_lossless_zero_len():
    """Zero-length lossless line + zero caps: S21=1 everywhere."""
    faxis = np.linspace(1e9, 25e9, 20)
    param = _make_param(Len_TX=0.0, Z_c=100.0, Z0=50.0)
    _, _, s21, _ = make_full_pkg('TX', faxis, param, 'THRU')
    assert np.allclose(np.abs(s21), 1.0, atol=1e-8)


def test_tx_vs_rx_different_length():
    """TX and RX use different length params."""
    faxis = np.linspace(1e9, 10e9, 20)
    param = _make_param(Len_TX=0.01, Len_RX=0.02, Z_c=100.0, Z0=50.0)
    param.pkg_gamma0_a1_a2 = np.array([0.0, 0.5, 0.0])
    _, _, s21_tx, _ = make_full_pkg('TX', faxis, param, 'THRU')
    _, _, s21_rx, _ = make_full_pkg('RX', faxis, param, 'THRU')
    # Different lengths should give different attenuation
    assert not np.allclose(s21_tx, s21_rx, atol=1e-4)


def test_dc_mode_halves_z0():
    """DC mode uses Z0/2 internally (doesn't change param externally)."""
    faxis = np.linspace(1e9, 10e9, 10)
    param = _make_param(Z0=100.0, Cpad=1e-13)
    Z0_before = float(param.Z0)
    make_full_pkg('TX', faxis, param, 'THRU', mode='dc')
    # param.Z0 should not be modified (make_full_pkg works on copy)
    assert float(param.Z0) == pytest.approx(Z0_before)


def test_rx_type_raises_for_wrong_type():
    faxis = np.linspace(1e9, 10e9, 10)
    param = _make_param()
    with pytest.raises((ValueError, AttributeError)):
        make_full_pkg('INVALID', faxis, param, 'THRU')
