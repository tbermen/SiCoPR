"""Verification tests for OptFom_Update_BEST_Post_Optimize().

# ============================================================
# MATLAB GROUND TRUTH (lines 3843-3890)
# Updates BEST after optimization: cursor value, CTLE gains,
# precursor/postcursor samples, DFE tap clipping.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Update_BEST_Post_Optimize.py_impl import OptFom_Update_BEST_Post_Optimize


def _f(N=50):
    return np.linspace(0, 50e9, N)


def _sbr(M=8, peak=40, N=200):
    sbr = np.zeros(N)
    sbr[peak] = 1.0
    for k in range(1, 4):
        sbr[peak + k * M] = 0.3 / k
    return sbr


def _param(M=8):
    return SimpleNamespace(
        samples_per_ui=M,
        ui=1 / 25e9,
        ndfe=3,
        ndfe_passed=3,
        CTLE_type='CL93',
        CTLE_fz=np.array([1e9]),
        CTLE_fp1=np.array([5e9]),
        CTLE_fp2=np.array([10e9]),
        Floating_DFE=False,
        fb=25e9,
        fb_BW_cutoff=1.0,
        BTorder=2,
        fb_BT_cutoff=1.0,
        RC_Start=5e9,
        RC_end=25e9,
    )


def _op():
    return SimpleNamespace(
        Bessel_Thomson=False,
        Butterworth=False,
        Raised_Cosine=False,
    )


def _best(M=8, cursor_i=40, N=200):
    return SimpleNamespace(
        sbr=_sbr(M=M, peak=cursor_i, N=N),
        cursor_i=cursor_i,
        ctle=0,
        gdc=-6.0,
        G_high_pass=0,
        bmax=np.array([0.9, 0.9, 0.9]),
        bmin=np.array([-0.9, -0.9, -0.9]),
    )


def test_cursor_value_set():
    """BEST.cursor = sbr at cursor_i."""
    B = _best(cursor_i=40)
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(), _op())
    assert result.cursor == pytest.approx(float(B.sbr[40]))


def test_h_r_is_array():
    """BEST.H_r is a numpy array of same length as f."""
    B = _best()
    f = _f(50)
    result = OptFom_Update_BEST_Post_Optimize(B, f, _param(), _op())
    assert len(result.H_r) == len(f)


def test_ctle_gain_is_complex_array():
    """BEST.ctle_gain is complex-valued."""
    B = _best()
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(), _op())
    assert np.iscomplexobj(result.ctle_gain)


def test_precursors_before_cursor():
    """sampled_sbr_precursors are the sbr samples before cursor."""
    B = _best(cursor_i=40)
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(), _op())
    np.testing.assert_array_equal(result.sampled_sbr_precursors, B.sbr[:40])


def test_postcursors_correct_length():
    """sampled_sbr_postcursors has the right number of elements (UI-spaced after cursor)."""
    M = 8
    cursor_i = 40
    N = 200
    B = _best(M=M, cursor_i=cursor_i, N=N)
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(M), _op())
    expected_count = len(range(cursor_i + M, N, M))
    assert len(result.sampled_sbr_postcursors) == expected_count
