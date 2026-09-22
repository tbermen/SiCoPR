"""Verification tests for add_brdorig().

# ============================================================
# MATLAB GROUND TRUTH (lines 4821-4841)
# switch chdata.type: 'THRU'→z_bp_tx; 'NEXT'→z_bp_next; 'FEXT'→z_bp_fext
# synth_tline for TX and RX board traces
# include_pcb=1: tx→ch→rx cascade (three stages)
# include_pcb=2: ch→rx cascade (two stages)
# Returns (s11, s12, s21, s22) as 1-D arrays of length N.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.add_brdorig.py_impl import add_brdorig


def _param():
    return SimpleNamespace(
        z_bp_tx=0.05, z_bp_next=0.04, z_bp_fext=0.03, z_bp_rx=0.06,
        brd_Z_c=np.array([50.0, 50.0]),
        Z0=50.0,
        brd_gamma0_a1_a2=np.array([0.0, 0.01, 0.001]),
        brd_tau=0.0,
    )


def _chdata(chtype='THRU', N=10):
    f = np.linspace(1e9, 20e9, N)
    s0 = np.zeros(N, dtype=complex)
    s1 = np.ones(N, dtype=complex)
    return SimpleNamespace(
        type=chtype, faxis=f,
        sdd11_raw=s0, sdd12_raw=s1, sdd21_raw=s1, sdd22_raw=s0,
    )


def _op(include_pcb=1):
    return SimpleNamespace(include_pcb=include_pcb)


def test_thru_pcb1_output_length():
    """THRU + include_pcb=1: four 1-D arrays of length N."""
    N = 15
    s11, s12, s21, s22 = add_brdorig(_chdata('THRU', N), _param(), _op(1))
    for arr in (s11, s12, s21, s22):
        assert len(arr) == N


def test_next_pcb2_output_length():
    """NEXT + include_pcb=2: uses z_bp_next, still correct length."""
    N = 10
    s11, s12, s21, s22 = add_brdorig(_chdata('NEXT', N), _param(), _op(2))
    assert len(s21) == N


def test_fext_uses_z_bp_fext():
    """FEXT type selects z_bp_fext; no error."""
    N = 8
    s11, s12, s21, s22 = add_brdorig(_chdata('FEXT', N), _param(), _op(1))
    assert len(s21) == N


def test_unknown_type_raises():
    """Unknown chdata.type raises ValueError."""
    with pytest.raises(ValueError, match='Unknown chdata.type'):
        add_brdorig(_chdata('XMIT'), _param(), _op(1))


def test_pcb1_vs_pcb2_differ():
    """include_pcb=1 and =2 produce different S21 (different cascade stages)."""
    cd = _chdata('THRU')
    _, _, s21_1, _ = add_brdorig(cd, _param(), _op(1))
    _, _, s21_2, _ = add_brdorig(cd, _param(), _op(2))
    assert not np.allclose(s21_1, s21_2)


# ============================================================
# COM Octave oracle — add_brdorig run under Octave with synth_tline and
# combines4p verbatim from the reference. Board values are a 93A-style set:
#
#   param.z_bp_tx=151  z_bp_next=72  z_bp_fext=53  z_bp_rx=73
#   param.brd_Z_c=[109.8 109.8]  Z0=50
#   param.brd_gamma0_a1_a2=[0 4.114e-4 2.547e-4]  brd_tau=6.191e-3
#   chdata.faxis = [0 10 20 30] GHz
#
# Pinned to 1e-15, not to the bit: the residual against Octave is ~2e-16 and
# comes from complex division, in combines4p and in synth_tline's
# (1-rho^2)*e./(1-rho^2*e.^2). Everything upstream of those divides —
# gamma, exp(-d*gamma), rho — is bit-identical, so the form is right and the
# last bit is the division kernel (numpy's Smith form against libstdc++'s).
# ============================================================
_F = np.array([0.0, 10e9, 20e9, 30e9])
_S11 = np.array([6.1507667874128715e-05 - 0.022733539258586129j,
                 0.014937276875423495 - 0.049582327749823123j,
                 -0.013706892768110879 + 0.0030071801298719243j,
                 -0.044529591937863711 + 0.067010762277726676j])
_S21 = np.array([0.80000000000000004 + 0j,
                 -0.52173825335165092 - 0.26826660762634308j,
                 0.21720663621409037 + 0.30364297285189656j,
                 -0.02328000540937817 - 0.15829731945974113j])
_S22 = np.array([-0.019688260742053185 + 0.0042165699599159426j,
                 -0.024818995992797618 - 0.037218721788328184j,
                 0.01959368200740793 - 0.0011700728985309396j,
                 0.01427548032640243 + 0.027812127778331513j])


def _oracle_param():
    return SimpleNamespace(
        z_bp_tx=151.0, z_bp_next=72.0, z_bp_fext=53.0, z_bp_rx=73.0,
        brd_Z_c=np.array([109.8, 109.8]), Z0=50.0,
        brd_gamma0_a1_a2=np.array([0.0, 4.114e-4, 2.547e-4]), brd_tau=6.191e-3)


def _oracle_chdata(chtype):
    return SimpleNamespace(type=chtype, faxis=_F.copy(), sdd11_raw=_S11.copy(),
                           sdd12_raw=_S21.copy(), sdd21_raw=_S21.copy(),
                           sdd22_raw=_S22.copy())


@pytest.mark.parametrize('chtype,ipcb,o11,o21', [
    ('THRU', 1,
     [6.1507667874128715e-05 - 0.022733539258586129j,
      0.060777703377537795 + 0.0081073832012307848j,
      0.039396113547284714 + 0.0016674103241923841j,
      0.052445532572417478 - 0.00013293185621924457j],
     [0.80000000000000004 + 0j,
      0.06738513883815922 - 0.23753970986729778j,
      -0.02604385230777332 - 0.074286130857829494j,
      -0.011590670344005299 - 0.01289937524304987j]),
    ('NEXT', 1,
     [6.1507667874128715e-05 - 0.022733539258586129j,
      0.056654354925143861 - 0.028119889881102634j,
      0.057514317242730061 - 0.019645663074883135j,
      0.073901030450983035 - 0.0030377463099390454j],
     [0.80000000000000004 + 0j,
      -0.18887779952119241 - 0.27630681439578542j,
      -0.098718915478802099 + 0.093575634724762616j,
      0.026514450614569991 + 0.02701838232752065j]),
    ('FEXT', 1,
     [6.1507667874128715e-05 - 0.022733539258586129j,
      0.068444113097600873 + 0.026798371809033685j,
      0.017569516478703067 + 0.0023144678827492753j,
      0.084860932796173433 - 0.011432340297193359j],
     [0.80000000000000004 + 0j,
      0.15952698071344251 - 0.32259302760454395j,
      -0.040685543998474495 - 0.15004736549412931j,
      -0.034774097200549837 - 0.029592037127320059j]),
    ('NEXT', 2,
     [6.1507667874128715e-05 - 0.022733539258586129j,
      0.019822117152713932 - 0.044506432568581035j,
      -0.013832947205436185 + 0.0078774772512478376j,
      -0.045539376386581398 + 0.067600362751255069j],
     [0.80000000000000004 + 0j,
      0.38164877017998855 + 0.22355390104816492j,
      0.064288707599198719 + 0.21545187091398701j,
      -0.039410198144872136 + 0.066863606586729393j]),
])
def test_octave_oracle_cascade(chtype, ipcb, o11, o21):
    s11, s12, s21, s22 = add_brdorig(_oracle_chdata(chtype), _oracle_param(), _op(ipcb))
    np.testing.assert_allclose(s11, o11, rtol=0, atol=1e-15)
    np.testing.assert_allclose(s21, o21, rtol=0, atol=1e-15)
    np.testing.assert_allclose(s12, o21, rtol=0, atol=1e-15)


def test_octave_oracle_thru_s22():
    _, _, _, s22 = add_brdorig(_oracle_chdata('THRU'), _oracle_param(), _op(1))
    np.testing.assert_allclose(s22, [
        -0.019688260742053185 + 0.0042165699599159426j,
        0.015493296267463295 - 0.016078608834283793j,
        0.037457276816504417 - 0.0052727301648360899j,
        0.038545254215910901 - 0.0063262547858913276j], rtol=0, atol=1e-15)


# ============================================================
# COM Octave oracle — include_pcb is matched, not truncated.
# MATLAB's `switch` tests equality, so a value that is neither 1 nor 2 leaves
# every output unassigned. COM Octave, OP.include_pcb = 1.5:
#     error: element number 1 undefined in return list
# ============================================================
@pytest.mark.parametrize('bad', [0, 1.5, 2.5, 3, -1])
def test_include_pcb_must_match_exactly(bad):
    with pytest.raises(ValueError, match='include_pcb'):
        add_brdorig(_chdata('THRU'), _param(), _op(bad))


def test_include_pcb_accepts_float_and_bool_one():
    """MATLAB's switch matches 1.0 and logical true against `case 1`."""
    ref = add_brdorig(_chdata('THRU'), _param(), _op(1))
    for spelling in (1.0, True, np.float64(1.0)):
        got = add_brdorig(_chdata('THRU'), _param(), _op(spelling))
        np.testing.assert_array_equal(got[2], ref[2])
