"""Tests for s21_pkg.

MATLAB GROUND TRUTH:
  INC_PACKAGE=0: returns s21 unchanged with warning.
  With matched terminations (gamma_tx=gamma_rx=0): VTF=s21.
  IDEAL_TX_TERM=True, IDEAL_RX_TERM=True: gamma_tx=0, gamma_rx=0 → VTF=s21.
  SCH.Parameters has shape (2,2,nfreq).
  sigma_ACCM_at_tp0=0 for non-DC mode.
  RX_CALIBRATION=1, channel_number=2: only RX pkg applied.
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.s21_pkg.py_impl import s21_pkg, _bessel, _Bessel_Thomson_Filter


def _make_chdata(faxis, s21=None, s11=None, s22=None, s12=None, chtype='THRU', mode='dd'):
    nf = len(faxis)
    ch = SimpleNamespace()
    ch.faxis = np.asarray(faxis, dtype=float)
    ch.type = chtype
    s21 = np.ones(nf, dtype=complex) if s21 is None else np.asarray(s21, dtype=complex)
    s11 = np.zeros(nf, dtype=complex) if s11 is None else np.asarray(s11, dtype=complex)
    s22 = np.zeros(nf, dtype=complex) if s22 is None else np.asarray(s22, dtype=complex)
    s12 = s21.copy() if s12 is None else np.asarray(s12, dtype=complex)
    setattr(ch, f's{mode}21_raw', s21)
    setattr(ch, f's{mode}11_raw', s11)
    setattr(ch, f's{mode}22_raw', s22)
    setattr(ch, f's{mode}12_raw', s12)
    return ch


def _make_param(Z0=100.0, R_diepad=None, kappa1=1.0, kappa2=1.0):
    p = SimpleNamespace()
    p.Z0 = Z0
    p.R_diepad = np.array([Z0, Z0]) if R_diepad is None else np.asarray(R_diepad, dtype=float)
    p.Tx_rd_sel = 1
    p.Rx_rd_sel = 2
    p.kappa1 = kappa1
    p.kappa2 = kappa2
    p.C_diepad = np.array([0.0, 0.0])
    p.C_pkg_board = np.array([0.0, 0.0])
    p.L_comp = np.array([0.0, 0.0])
    p.C_bump = np.array([0.0, 0.0])
    p.z_p_next_cases = np.array([[Z0]])
    p.pkg_Z_c = np.array([Z0, Z0])
    p.Pkg_len_TX = np.array([0.0])
    p.Pkg_len_RX = np.array([0.0])
    p.Pkg_len_NEXT = np.array([0.0])
    p.Pkg_len_FEXT = np.array([0.0])
    p.pkg_tau = 0.0
    p.pkg_gamma0_a1_a2 = np.array([0.0, 0.0, 0.0])
    p.PKG_NAME = None
    p.num_s4p_files = 1
    p.BTorder = 4
    p.fb_BT_cutoff = 0.75
    p.fb = 26.5625e9
    p.ACCM_MAX_Freq = 40e9
    p.AC_CM_RMS_TX = 0.0
    return p


def _make_OP(inc_pkg=1, ideal_tx=True, ideal_rx=True, rx_cal=0):
    op = SimpleNamespace()
    op.INC_PACKAGE = inc_pkg
    op.IDEAL_TX_TERM = ideal_tx
    op.IDEAL_RX_TERM = ideal_rx
    op.RX_CALIBRATION = rx_cal
    op.PSDRXCAL = False
    op.include_pcb = 0
    op.T_r_filter_type = 0
    op.transmitter_transition_time = 0.0
    op.T_r_meas_point = 0
    return op


def test_bessel_poly_dc_gain_unity():
    """Bessel polynomial filter DC gain = 1."""
    a = _bessel(4)
    param = SimpleNamespace(BTorder=4, fb_BT_cutoff=0.75, fb=26.5625e9)
    f = np.array([0.0, 1e9, 5e9])
    H = _Bessel_Thomson_Filter(param, f, True)
    assert abs(H[0]) == pytest.approx(1.0, abs=1e-6)


def test_inc_package_zero_returns_s21():
    """INC_PACKAGE=0: s21p = s21."""
    faxis = np.linspace(1e9, 25e9, 20)
    s21 = 0.8 * np.exp(-1j * 2 * np.pi * faxis * 100e-12)
    ch = _make_chdata(faxis, s21=s21)
    op = _make_OP(inc_pkg=0)
    s21p, SCH, sigma = s21_pkg(ch, _make_param(), op, 1)
    assert np.allclose(s21p, s21)


def test_ideal_term_vtf_equals_s21():
    """Ideal terminations (gamma_tx=gamma_rx=0) with zero-length pkg: VTF=s21."""
    faxis = np.linspace(1e9, 25e9, 20)
    s21 = 0.7 * np.exp(-1j * 2 * np.pi * faxis * 100e-12)
    ch = _make_chdata(faxis, s21=s21)
    op = _make_OP(ideal_tx=True, ideal_rx=True)
    s21p, SCH, sigma = s21_pkg(ch, _make_param(), op, 1)
    assert np.allclose(np.abs(s21p), np.abs(s21), atol=1e-6)


def test_sch_parameters_shape():
    """SCH.Parameters.shape = (2, 2, nfreq)."""
    faxis = np.linspace(1e9, 10e9, 15)
    ch = _make_chdata(faxis)
    op = _make_OP()
    _, SCH, _ = s21_pkg(ch, _make_param(), op, 1)
    assert SCH.Parameters.shape == (2, 2, 15)


def test_sigma_accm_zero_for_dd_mode():
    """sigma_ACCM_at_tp0 = 0 for DD mode."""
    faxis = np.linspace(1e9, 25e9, 30)
    ch = _make_chdata(faxis)
    op = _make_OP()
    _, _, sigma = s21_pkg(ch, _make_param(), op, 1, mode='dd')
    assert sigma == pytest.approx(0.0)


def test_sch_impedance_dd_mode():
    """SCH.Impedance = Z0*2 for DD mode."""
    faxis = np.linspace(1e9, 10e9, 10)
    ch = _make_chdata(faxis)
    _, SCH, _ = s21_pkg(ch, _make_param(Z0=100.0), _make_OP(), 1, mode='dd')
    assert SCH.Impedance == pytest.approx(200.0)


def test_rx_calibration_channel2():
    """RX_CALIBRATION=1, channel=2: applies only RX pkg."""
    faxis = np.linspace(1e9, 10e9, 10)
    ch = _make_chdata(faxis)
    op = _make_OP(rx_cal=1, ideal_tx=False, ideal_rx=False)
    s21p, SCH, sigma = s21_pkg(ch, _make_param(), op, 2)
    assert len(s21p) == 10
    assert SCH.Parameters.shape == (2, 2, 10)
