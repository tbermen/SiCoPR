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
