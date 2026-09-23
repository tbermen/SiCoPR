"""Verification tests for OptFom_Compute_CTLE().

# ============================================================
# MATLAB GROUND TRUTH (lines 3144-3218)
# Returns (chdata, H_ctf, H_low_xc, H_ctf2).
# H_ctf = H_low * ctle_gain; CL93: H_low=1.
# INCLUDE_CTLE=1: applies TD_CTLE to each channel.
# INCLUDE_CTLE=0: copies uneq → ctle field unchanged.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Compute_CTLE.py_impl import OptFom_Compute_CTLE


def _param(M=8, fb=25e9):
    return SimpleNamespace(
        fb=fb, ui=1.0/fb, samples_per_ui=M,
        CTLE_type='CL93',
        ctle_gdc_values=np.array([-6.0, 0.0]),
        CTLE_fp1=np.array([10e9, 12e9]),
        CTLE_fp2=np.array([20e9, 25e9]),
        CTLE_fz=np.array([5e9, 6e9]),
        g_DC_HP_values=np.array([0.0]),
        f_HP=np.array([1e9]),
        f_HP_Z=np.array([1e9, 1.2e9]),
        f_HP_P=np.array([2e9, 2.4e9]),
        num_s4p_files=1,
    )


def _op(include_ctle=1, tdmode=True):
    return SimpleNamespace(
        TDMODE=tdmode,
        INCLUDE_CTLE=include_ctle,
        impulse_response_truncation_threshold=1e-3,
        RX_CALIBRATION=False,
    )


def _chdata(M=8, fb=25e9, N=200):
    f = np.linspace(0, fb/2, N)
    ir = np.zeros(N)
    ir[0] = 1.0
    sdd21 = np.ones(N, dtype=complex) * 0.9
    cd = SimpleNamespace(
        faxis=f,
        uneq_pulse_response=ir.copy(),
        uneq_imp_response=ir.copy(),
        sdd21=sdd21,
    )
    return [cd]


def _THIS(ctle_index=1, g_LP_index=1):
    return SimpleNamespace(ctle_index=ctle_index, g_LP_index=g_LP_index)


def test_returns_four_outputs():
    """OptFom_Compute_CTLE returns 4 values."""
    result = OptFom_Compute_CTLE(_chdata(), np.ones(200), _THIS(), np.zeros(512), _param(), _op())
    assert len(result) == 4


def test_H_ctf_shape():
    """H_ctf has same length as chdata faxis."""
    chdata = _chdata()
    chdata_out, H_ctf, _, _ = OptFom_Compute_CTLE(
        chdata, np.ones(200), _THIS(), np.zeros(512), _param(), _op())
    assert len(H_ctf) == len(chdata[0].faxis)


def test_ctle_field_set_when_include_ctle():
    """ctle_pulse_response set on chdata when INCLUDE_CTLE=1."""
    chdata = _chdata()
    chdata_out, _, _, _ = OptFom_Compute_CTLE(
        chdata, np.ones(200), _THIS(), np.zeros(512), _param(), _op(include_ctle=1))
    assert hasattr(chdata_out[0], 'ctle_pulse_response')


def test_ctle_field_equals_uneq_when_no_ctle():
    """When INCLUDE_CTLE=0, ctle_field equals uneq_field."""
    chdata = _chdata()
    chdata_out, _, _, _ = OptFom_Compute_CTLE(
        chdata, np.ones(200), _THIS(), np.zeros(512), _param(), _op(include_ctle=0))
    uneq = np.asarray(chdata_out[0].uneq_pulse_response)
    ctle = np.asarray(chdata_out[0].ctle_pulse_response)
    assert np.allclose(uneq, ctle)


def test_sdd21ctf_set():
    """sdd21ctf is set on chdata[0]."""
    chdata = _chdata()
    chdata_out, _, _, _ = OptFom_Compute_CTLE(
        chdata, np.ones(200), _THIS(), np.zeros(512), _param(), _op())
    assert hasattr(chdata_out[0], 'sdd21ctf')


def test_H_ctf2_is_one_no_calibration():
    """H_ctf2=1 when RX_CALIBRATION=False."""
    chdata = _chdata()
    _, _, _, H_ctf2 = OptFom_Compute_CTLE(
        chdata, np.ones(200), _THIS(), np.zeros(512), _param(), _op())
    assert H_ctf2 == 1


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py; OptFom_Compute_CTLE,
# OptFom_FD_or_TD_Fields, FD_CTLE and TD_CTLE are byte-identical between
# octave/com_ieee8023_4p16p0_octave_compat.m and matlab/com_ieee8023_4p16p0.m).
#
# Fixture: f = linspace(0, 40 GHz, 21), two channels whose
#   uneq_pulse_response are the Gaussians below, ui = 1/53.125e9,
#   samples_per_ui = 8, ctle_index = g_LP_index = 2,
#   impulse_response_truncation_threshold = 1e-3.
#
# What these catch:
#   1. find(...,1,'last') is EMPTY when nothing clears the threshold, and
#      MATLAB's ir(1:[]) is EMPTY -- the port left the response untouched and
#      ran the CTLE over the whole thing.
#   2. max([]) is [] in MATLAB, so an empty response is not an error there.
#   3. For CL93, H_low_xc is the SCALAR 1; the port returned ones(len(f_xc)).
# ============================================================

_NF = 21
_F = np.linspace(0.0, 40e9, _NF)
_N = np.arange(64)
_IR0 = (np.exp(-((_N - 12.0) / 3.0) ** 2) * 0.8
        - 0.03 * np.exp(-((_N - 30.0) / 9.0) ** 2))
_IR1 = 0.2 * np.exp(-((_N - 15.0) / 4.0) ** 2)
_SDD21 = (0.9 ** (_F / 1e10)) * np.exp(-1j * 2 * np.pi * _F * 1e-10)
_CTLE_GAIN = np.exp(-1j * _F / 3e10) * (1 + _F / 1e11)
_F_XC = np.array([1e9, 5e9, 12e9])


def _oracle_param(ctle_type, nfiles=2):
    return SimpleNamespace(
        ui=1.0 / 53.125e9, samples_per_ui=8, num_s4p_files=nfiles,
        CTLE_type=ctle_type,
        ctle_gdc_values=np.array([0.0, -3.0, -6.0, -9.0]),
        CTLE_fp1=np.array([10e9, 11e9, 12e9, 13e9]),
        CTLE_fp2=np.array([20e9, 21e9, 22e9, 23e9]),
        CTLE_fz=np.array([5e9, 5.5e9, 6e9, 6.5e9]),
        g_DC_HP_values=np.array([0.0, -0.5, -1.0, -1.5]),
        f_HP=np.array([0.6e9, 0.7e9, 0.8e9, 0.9e9]),
        f_HP_Z=np.array([0.5e9, 0.6e9, 0.7e9, 0.8e9]),
        f_HP_P=np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9]))


def _oracle_chdata(irs=None):
    irs = irs if irs is not None else (_IR0, _IR1)
    return [SimpleNamespace(faxis=_F.copy(), sdd21=_SDD21.copy(),
                            uneq_pulse_response=np.asarray(ir, float).copy())
            for ir in irs]


def _run(ctle_type, irs=None, thr=1e-3, rxcal=False, nfiles=2):
    OP = SimpleNamespace(TDMODE=1, INCLUDE_CTLE=1, RX_CALIBRATION=rxcal,
                         impulse_response_truncation_threshold=thr)
    return OptFom_Compute_CTLE(_oracle_chdata(irs), _CTLE_GAIN,
                               SimpleNamespace(ctle_index=2, g_LP_index=2),
                               _F_XC, _oracle_param(ctle_type, nfiles), OP)


def test_octave_cl120d_nominal():
    """COM Octave, CTLE_type='CL120d', RX_CALIBRATION on."""
    cd, H_ctf, H_low_xc, H_ctf2 = _run('CL120d', rxcal=True)
    assert len(cd[0].uneq_pulse_response) == 48
    assert len(cd[1].uneq_pulse_response) == 26
    np.testing.assert_allclose(cd[0].ctle_pulse_response[:6], [
        -9.1595508822718506e-08, -9.094463221341232e-08,
        2.5792228872810747e-06, 2.851098851800785e-05,
        0.0002074241646813266, 0.0011764315839140878], rtol=1e-10)
    np.testing.assert_allclose(cd[0].ctle_pulse_response[-3:], [
        -0.0035061418279677052, -0.0013217687384532095,
        0.00026957226510512176], rtol=1e-10)
    np.testing.assert_allclose(cd[1].ctle_pulse_response[:4], [
        3.993488246279123e-08, 3.0997388654047767e-07,
        1.7634998152055569e-06, 8.7465253648031449e-06], rtol=1e-10)
    np.testing.assert_allclose(H_low_xc, [
        0.98160391233563937 + 0.026280125234801093j,
        0.99892466965006288 + 0.0076809310709795263j,
        0.99981029710969693 + 0.0032520495480534537j], rtol=1e-13)
    np.testing.assert_allclose(H_ctf[:3], [
        0.94406087628592339 + 0j,
        1.0127063761427937 - 0.049783464393739207j,
        1.0303690876692548 - 0.12823563726028703j], rtol=1e-13)
    np.testing.assert_allclose(H_ctf2[:3], [
        0.66834391756861466 + 0j,
        0.75665761442085155 + 0.16701986530625368j,
        0.90123913747554174 + 0.25076962966696148j], rtol=1e-13)


def test_octave_cl120e_nominal():
    """COM Octave, CTLE_type='CL120e'."""
    cd, _, H_low_xc, _ = _run('CL120e')
    np.testing.assert_allclose(cd[0].ctle_pulse_response[:4], [
        -1.6735764812934535e-07, -1.6504261461244308e-07,
        4.7148220324912863e-06, 5.206400347374425e-05], rtol=1e-10)
    np.testing.assert_allclose(H_low_xc, [
        1.3770739064856712 + 0.41478129713423839j,
        1.7948620119547249 + 0.1748696426300394j,
        1.8263893671234763 + 0.075752358652985352j], rtol=1e-13)


def test_octave_cl93_h_low_xc_is_the_scalar_one():
    """MATLAB sets H_low_xc=1 for CL93; f_xc has three points and is ignored.

    COM Octave: size(H_low_xc) is 1x1. The port returned ones(len(f_xc)).
    """
    cd, _, H_low_xc, _ = _run('CL93')
    assert np.ndim(H_low_xc) == 0
    assert H_low_xc == 1.0
    np.testing.assert_allclose(cd[0].ctle_pulse_response[:4], [
        -9.1621892327380673e-08, -9.1023338736036879e-08,
        2.5798616787234481e-06, 2.8520576436841092e-05], rtol=1e-10)


def test_octave_all_zero_response_truncates_to_empty():
    """An all-zero response clears no threshold, so find() is empty.

    COM Octave, chdata(1).uneq_pulse_response = zeros(1,64): both
    uneq_pulse_response and ctle_pulse_response come back 1x0, while
    channel 2 is untouched. Leaving ir alone kept all 64 samples.
    """
    cd, _, _, _ = _run('CL120d', irs=(np.zeros(64), _IR1))
    assert len(cd[0].uneq_pulse_response) == 0
    assert len(cd[0].ctle_pulse_response) == 0
    assert len(cd[1].uneq_pulse_response) == 26


def test_octave_threshold_of_one_truncates_every_channel_to_empty():
    """thr=1: nothing is strictly greater than the peak, so find() is empty.

    COM Octave: all four fields come back 1x0.
    """
    cd, _, _, _ = _run('CL120d', thr=1.0)
    for c in cd:
        assert len(c.uneq_pulse_response) == 0
        assert len(c.ctle_pulse_response) == 0


def test_octave_empty_response_is_not_an_error():
    """MATLAB max([]) is [] and `[] > []*thr` is empty, so an empty response
    reaches find() without erroring.

    COM Octave, chdata(1).uneq_pulse_response = []: 1x0 out, channel 2 still
    truncated to 26. np.max raised "zero-size array to reduction operation".
    """
    cd, _, _, _ = _run('CL120d', irs=(np.zeros(0), _IR1))
    assert len(cd[0].ctle_pulse_response) == 0
    assert len(cd[1].ctle_pulse_response) == 26
