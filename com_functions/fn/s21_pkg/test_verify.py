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


def test_inc_package_zero_has_no_second_output():
    """INC_PACKAGE=0 assigns s21p and nothing else, so a three-output call is
    an error in the reference, not an SCH invented here. COM Octave:
        error: element number 2 undefined in return list
    """
    faxis = np.linspace(1e9, 25e9, 20)
    s21 = 0.8 * np.exp(-1j * 2 * np.pi * faxis * 100e-12)
    ch = _make_chdata(faxis, s21=s21)
    op = _make_OP(inc_pkg=0)
    with pytest.raises(ValueError):
        s21_pkg(ch, _make_param(), op, 1)


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


# ============================================================
# COM Octave oracle values (2026-09-22)
# s21_pkg run verbatim from octave/com_ieee8023_4p16p0_octave_compat.m via
# tools/octave_oracle.py, with make_full_pkg, make_pkg, synth_tline,
# combines4p, Bessel_Thomson_Filter and bessel as its subfunctions, on the
# package below (12 mm, 78.2 ohm, real die C/L, gamma0_a1_a2 from the spec).
#
# Divergences these pin:
#   1. the INLINED _make_pkg used np.finfo(float).tiny where MATLAB's
#      `f(f<eps)=eps` means eps(1); the DC point then went into synth_tline's
#      sqrt and log 292 orders of magnitude too small. Relative agreement on
#      s21p went from 2.9e-14 to 6e-16 across every branch when fixed;
#   2. INC_PACKAGE=0 leaves SCH unassigned, so a three-output call is an
#      error, not an invented two-port;
#   3. the dc-mode AC CM integral had a guard MATLAB does not have:
#      ACCM_MAX_Freq=0 is NaN in the reference and was 0.0 here, and an
#      ACCM_MAX_Freq below faxis(1) is an f_int(0) subscript error.
# The VTF expression itself was checked separately against the reference's own
# SCH.Parameters and reproduces s21p to 1.1 ulp, 36 of 51 points exact.
# ============================================================

_NF_ORACLE = 51
_F_ORACLE = np.linspace(0.0, 50e9, _NF_ORACLE)
_ATT = np.exp(-np.sqrt(_F_ORACLE / 1e9) * 0.12)
_S21_ORACLE = _ATT * np.exp(-1j * 2 * np.pi * _F_ORACLE * 2e-10)
_S11_ORACLE = 0.05 * np.exp(-1j * 2 * np.pi * _F_ORACLE * 1e-11)
_S22_ORACLE = 0.04 * np.exp(-1j * 2 * np.pi * _F_ORACLE * 1.2e-11)
_SDC21_ORACLE = 0.01 * _ATT * np.exp(-1j * 2 * np.pi * _F_ORACLE * 2e-10)
_SDC11_ORACLE = 0.006 * np.ones(_NF_ORACLE, dtype=complex)
_IDX_ORACLE = [0, 1, 10, 25, 50]


def _oracle_chdata():
    c = SimpleNamespace(faxis=_F_ORACLE, type='THRU')
    for nm, v in [('sdd21_raw', _S21_ORACLE), ('sdd12_raw', _S21_ORACLE),
                  ('sdd11_raw', _S11_ORACLE), ('sdd22_raw', _S22_ORACLE),
                  ('sdc21_raw', _SDC21_ORACLE), ('sdc12_raw', _SDC21_ORACLE),
                  ('sdc11_raw', _SDC11_ORACLE), ('sdc22_raw', _SDC11_ORACLE),
                  ('scd21_raw', _SDC21_ORACLE), ('scd12_raw', _SDC21_ORACLE),
                  ('scd11_raw', _SDC11_ORACLE), ('scd22_raw', _SDC11_ORACLE)]:
        setattr(c, nm, v)
    return c


def _oracle_param(**over):
    p = SimpleNamespace(
        Z0=50.0, R_diepad=np.array([55.0, 48.0]), Tx_rd_sel=1, Rx_rd_sel=2,
        kappa1=1.0, kappa2=1.0, C_diepad=np.array([1.3e-13, 1.1e-13]),
        C_pkg_board=np.array([5e-14, 5e-14]),
        L_comp=np.array([1e-10, 1.2e-10]), C_bump=np.array([3e-14, 3.2e-14]),
        z_p_next_cases=np.array([[12.0]]), pkg_Z_c=np.array([78.2, 78.2]),
        Pkg_len_TX=np.array([12.0]), Pkg_len_RX=np.array([12.0]),
        Pkg_len_NEXT=np.array([12.0]), Pkg_len_FEXT=np.array([12.0]),
        pkg_tau=6.141e-3,
        pkg_gamma0_a1_a2=np.array([0.0, 1.734e-3, 1.455e-4]),
        num_s4p_files=4, BTorder=4, fb_BT_cutoff=0.75, fb=53.125e9,
        ACCM_MAX_Freq=30e9, AC_CM_RMS_TX=0.01, C_v=np.array([2e-14, 2e-14]),
        PKG_NAME=None)
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _oracle_OP(**over):
    o = SimpleNamespace(INC_PACKAGE=1, IDEAL_TX_TERM=0, IDEAL_RX_TERM=0,
                        RX_CALIBRATION=0, PSDRXCAL=0, include_pcb=0,
                        T_r_filter_type=0, transmitter_transition_time=0.008,
                        T_r_meas_point=0, TX_BesselThomson=1)
    for k, v in over.items():
        setattr(o, k, v)
    return o


@pytest.mark.parametrize('op_over,ch,mode,die,s21p,sigma,impedance', [
    ({}, 1, 'dd', 1,
     [0.93349980746564554 - 1.918383411540967e-14j,
      -0.54406064430324075 - 0.55931010481741372j,
      -0.43182555571111414 + 0.28552398970753889j,
      0.31634600827236686 + 0.026790255760077108j,
      0.15302813231797166 + 0.049260381028601162j], 0.0, 100.0),
    ({}, 1, 'dd', 0,
     [0.99999999999997957 - 2.0425195785897526e-14j,
      -0.54350364692831243 - 0.63813137115993368j,
      -0.54713583614416772 + 0.010432202827338779j,
      -0.034932465935515189 + 0.39432504369499527j,
      -0.21264989809666454 - 0.083925650322607945j], 0.0, 100.0),
    (dict(T_r_filter_type=1, T_r_meas_point=1), 1, 'dd', 1,
     [0.93349980746564554 - 1.918383411540967e-14j,
      -0.5660098634152616 - 0.53695659491808911j,
      -0.28260016885227568 + 0.42663537605071195j,
      0.17892330260096032 - 0.23467003187742755j,
      -0.014382932899140288 - 0.11705290238959207j], 0.0, 100.0),
    (dict(PSDRXCAL=1), 4, 'dd', 1,
     [0.93349980746565508 - 9.550227203563991e-15j,
      -0.17074960355868254 - 0.78364827908115964j,
      0.16184791506613258 + 0.55135695700830156j,
      0.41506274811055305 + 0.026737154329357853j,
      0.26170229117086341 + 0.042522171603176617j], 0.0, 100.0),
    (dict(RX_CALIBRATION=1), 2, 'dd', 1,
     [0.97879282218596075 - 9.9192798756286193e-15j,
      -0.18338246484994297 - 0.81903326240456975j,
      0.17307586570401706 + 0.57902505417986649j,
      0.42417178500685621 + 0.018301253839163877j,
      0.26495267200706513 + 0.048494505068524638j], 0.0, 100.0),
    ({}, 1, 'dc', 1,
     [0.0093309689072829539 - 2.4483657420075963e-16j,
      -0.0055612001721685725 - 0.0038070636740024691j,
      -0.0019371882614887287 + 0.0023106462745561703j,
      0.0015552281311901224 + 8.1992542893171292e-05j,
      0.00063619985811761413 + 4.0701083542238064e-05j],
     0.008029571278976872, 25.0),
    ({}, 1, 'cd', 1,
     [0.0093309689072829435 - 2.5507097065507581e-16j,
      -0.005485781013628465 - 0.0035227120991023837j,
      -0.0020450293307475681 + 0.0020906067960137407j,
      0.0016592797335061482 + 0.0001612146619710823j,
      0.00073334795969766549 + 8.3548715856044452e-05j], 0.0, 25.0),
])
def test_octave_s21p_branches(op_over, ch, mode, die, s21p, sigma, impedance):
    """COM Octave s21p at faxis indices 0, 1, 10, 25 and 50, plus
    sigma_ACCM_at_tp0 and SCH.Impedance, for each branch of the function."""
    got, SCH, got_sigma = s21_pkg(_oracle_chdata(), _oracle_param(),
                                  _oracle_OP(**op_over), ch, mode, die)
    got = np.asarray(got).ravel()[_IDX_ORACLE]
    np.testing.assert_allclose(got, s21p, rtol=5e-15, atol=1e-16)
    assert got_sigma == pytest.approx(sigma, rel=1e-13, abs=1e-18)
    assert float(SCH.Impedance) == pytest.approx(impedance)


def test_octave_dc_sigma_accm_narrow_band():
    """COM Octave, dc mode, ACCM_MAX_Freq = 1e9:
        sigma_ACCM_at_tp0 = 0.011289650331314377"""
    _, _, sigma = s21_pkg(_oracle_chdata(), _oracle_param(ACCM_MAX_Freq=1e9),
                          _oracle_OP(), 1, 'dc', 1)
    assert sigma == pytest.approx(0.011289650331314377, rel=1e-13)


def test_octave_dc_sigma_accm_at_dc_only_is_nan():
    """COM Octave, dc mode, ACCM_MAX_Freq = 0: f_int is [0], the sum is empty
    and 0/0 gives NaN. The port's guard returned 0.0."""
    _, _, sigma = s21_pkg(_oracle_chdata(), _oracle_param(ACCM_MAX_Freq=0.0),
                          _oracle_OP(), 1, 'dc', 1)
    assert np.isnan(sigma)


def test_octave_dc_sigma_accm_below_first_point_errors():
    """COM Octave, dc mode, ACCM_MAX_Freq below faxis(1):
        error: f_int(0): subscripts must be either integers
               1 to (2^63)-1 or logicals"""
    with pytest.raises(IndexError):
        s21_pkg(_oracle_chdata(), _oracle_param(ACCM_MAX_Freq=-1.0),
                _oracle_OP(), 1, 'dc', 1)
