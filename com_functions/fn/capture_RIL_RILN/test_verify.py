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


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-22): capture_RIL_RILN run verbatim under Octave
# from octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m for this function).
#
# Two divergences it found:
#
# 1. The "odd case" branch.  MATLAB error()s when neither solution of the
#    quadratic is the one with Re(Z) > 0 (L5211).  The port instead picked the
#    solution with the smaller |rho| and carried on, so every number after that
#    point was unverifiable -- for S11=S22=0 it was literally NaN.
#
# 2. The dB formulas carried `+ np.finfo(float).eps` that MATLAB does not have.
#    d(20*log10 x)/dx = 20/(x ln10), so the epsilon moved ordinary values too,
#    not only the zeros: on the lossy 2-port below the RIL_dB gap was 4.44e-15
#    dB.  At a true zero it replaced the reference's answer outright.
#
# Complex arithmetic elsewhere is pinned to a tolerance, not bit-exactly:
# numpy's complex multiply/divide uses SIMD FMA where Octave's does not, which
# moves shared results by ~1 ulp on about half of all elements (measured
# 1758/4000 on complex a.*b and 1699/4000 on a./b, against 0/4000 on real a.*b
# and on real a.*b-c.*d).
# ---------------------------------------------------------------------------
_OCT_FREQ = np.array([1000000000, 10666666666.666666, 20333333333.333332,
                      30000000000])
_OCT_S11 = np.array([0.015450849718747373 - 0.047552825814757678j,
                     0.033456530317942886 - 0.037157241273869733j,
                     0.045677272882130089 - 0.02033683215378991j,
                     0.050000000000000003 + 7.347880794884119e-17j])
_OCT_S21 = np.array([0.63651609569945999 - 0.46245601328558789j,
                     0.61180390074648106 - 0.27239264639599015j,
                     0.55759234977696137 - 0.11851991268209859j,
                     0.48522452777010677 + 3.5653719888086831e-16j])
_OCT_S22 = np.array([0.012360679774997899 - 0.038042260651806145j,
                     0.026765224254354309 - 0.029725793019095788j,
                     0.03654181830570407 - 0.01626946572303193j,
                     0.040000000000000001 + 5.8783046359072947e-17j])


def _oracle_chdata():
    return _make_chdata(_OCT_FREQ, _OCT_S11, _OCT_S21.copy(), _OCT_S21.copy(),
                        _OCT_S22)


@pytest.mark.parametrize('field,expect', [
    ('RIL', [0.56168168974339761 - 0.55540430686018238j,
             0.58668014313497752 - 0.32757703494196688j,
             0.55347768457367996 - 0.14335624933573748j,
             0.48652409075548608 + 4.3030679633741315e-16j]),
    ('RILN', [-0.074834405956062389 - 0.092948293574594498j,
              -0.025123757611503539 - 0.055184388545976737j,
              -0.0041146652032814091 - 0.024836336653638894j,
              0.00129956298537931 + 7.3769597456544845e-17j]),
    ('Z_port1', [112.5873669756283 + 17.645942750394333j,
                 113.26211773414101 + 10.625835173572678j,
                 113.93360853817421 + 5.211025553804113j,
                 113.43498001687789 - 1.7725908067841279e-14j]),
    ('Z_port2', [114.28380710877825 + 14.316544969992064j,
                 112.83561001372553 + 8.4610234246577232j,
                 112.48891232333312 + 4.1122488393590846j,
                 111.61044934842329 - 1.3940060844736558e-14j]),
    ('rho_port1', [0.06564794495968998 + 0.077556456465471851j,
                   0.064509342447744561 + 0.046611042032131345j,
                   0.065684873462240834 + 0.022758182003110607j,
                   0.062946476795019607 - 7.7822878919213585e-17j]),
    ('rho_port2', [0.070806029050687502 + 0.062080506457439975j,
                   0.061790337388825223 + 0.037297395544295626j,
                   0.059126803726903321 + 0.018208501643938495j,
                   0.054867088956019917 - 6.2261624257612472e-17j]),
])
def test_octave_lossy_2port_values(field, expect):
    """COM Octave values for an ordinary lossy, slightly reflective 2-port."""
    rs = capture_RIL_RILN(_oracle_chdata())
    np.testing.assert_allclose(np.asarray(getattr(rs, field)).ravel(),
                               np.array(expect), rtol=2e-14, atol=1e-16)


def test_octave_dB_of_a_lossy_channel_has_no_epsilon_floor():
    """COM Octave, |RIL| ~ 1e-9 to 1e-11, where the epsilon is worth ~1e-4 dB.

    On the ordinary 2-port above the epsilon moved RIL_dB by only ~3e-15 dB,
    which the ~1e-15 dB of complex-FMA noise hides.  Drive |RIL| down and the
    two separate by ten orders of magnitude:
        S21 = S12 = [1e-9, 1e-10, 1e-11], S11 = 0.05, S22 = 0.04
        Octave RIL_dB  = [-179.98217476697135, -199.98217476697135,
                          -219.98217476697135]
        `abs(RIL)+eps`  = [-179.98217284227056, -199.98215551998271,
                          -219.9819822990043]     (off by up to 1.9e-4 dB)
        Octave RILN_dB = -0.017825233028645471 at all three."""
    freq = np.array([1e9, 10e9, 30e9])
    s21 = np.array([1e-9 + 0j, 1e-10 + 0j, 1e-11 + 0j])
    chdata = _make_chdata(freq, 0.05 * np.ones(3, dtype=complex), s21.copy(),
                          s21.copy(), 0.04 * np.ones(3, dtype=complex))
    rs = capture_RIL_RILN(chdata)
    np.testing.assert_allclose(
        np.asarray(rs.RIL_dB).ravel(),
        np.array([-179.98217476697135, -199.98217476697135,
                  -219.98217476697135]), rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(rs.RILN_dB).ravel(),
        np.full(3, -0.017825233028645471), rtol=0, atol=1e-13)


def test_octave_zero_s21_gives_minus_inf_not_minus_313():
    """COM Octave, S21=S12=0: RIL_dB = -Inf and RILN_dB = NaN.

    The epsilon floor answered -313.07119549054045 dB and 0 dB instead -- a
    finite number where the reference has none."""
    freq = np.array([1e9, 3e9, 10e9])
    z = np.zeros(3, dtype=complex)
    chdata = _make_chdata(freq, 0.05 * np.ones(3, dtype=complex), z, z,
                          0.04 * np.ones(3, dtype=complex))
    rs = capture_RIL_RILN(chdata)
    assert np.all(np.isneginf(rs.RIL_dB))
    assert np.all(np.isnan(rs.RILN_dB))


@pytest.mark.parametrize('s11v,s21v', [
    (0.0 + 0j, 0.8 + 0j),          # a = 0: the quadratic degenerates
    (0.5 + 0j, 0.8660254037844386 + 0j),   # lossless, both roots on one side
])
def test_octave_odd_case_raises(s11v, s21v):
    """COM Octave: "error: An odd case has occured. Please contact the tool
    developer."  The port used to return numbers (NaN, in the a=0 case)."""
    freq = np.array([1e9, 3e9, 10e9])
    s11 = s11v * np.ones(3, dtype=complex)
    s21 = s21v * np.ones(3, dtype=complex)
    chdata = _make_chdata(freq, s11, s21.copy(), s21.copy(), s11.copy())
    with pytest.raises(ValueError, match='odd case'):
        capture_RIL_RILN(chdata)
