"""Verification tests for OptFom_Calculate_Settings().

# ============================================================
# MATLAB GROUND TRUTH (lines 3034-3143)
# Computes H_r, f_xc, H_r_xc, H_sy, qual, Peak_Search_Range,
# phase_memory from chdata and param.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Calculate_Settings.py_impl import OptFom_Calculate_Settings


def _param(M=8, fb=25e9):
    return SimpleNamespace(
        fb=fb, ui=1.0/fb, samples_per_ui=M,
        fb_BW_cutoff=1.0, BTorder=2, fb_BT_cutoff=1.0,
        RC_Start=5e9, RC_end=20e9,
        ctle_gdc_values=np.array([-6.0, 0.0]),
        g_DC_HP_values=np.array([0.0]),
        gqual=None, g2qual=None,
        CTLE_type='CL93',
        cursor_index=1,
    )


def _op():
    return SimpleNamespace(
        Bessel_Thomson=False, Butterworth=False, Raised_Cosine=False,
        USE_ETA0_PSD=False, TDMODE=True, RxFFE=False,
    )


def _chdata(M=8, fb=25e9, N=512):
    f = np.linspace(0, fb/2, N)
    ir = np.zeros(N)
    ir[50] = 1.0
    cd = SimpleNamespace(
        faxis=f,
        uneq_pulse_response=ir,
        uneq_imp_response=ir,
    )
    return [cd]


def test_returns_settings_namespace():
    """OptFom_Calculate_Settings returns a SimpleNamespace."""
    p = _param()
    txffe = np.array([[0.8]])
    S = OptFom_Calculate_Settings(txffe, _chdata(), p, _op())
    assert hasattr(S, 'H_r')


def test_H_r_shape():
    """H_r has same length as chdata faxis."""
    p = _param()
    chdata = _chdata()
    txffe = np.array([[0.8]])
    S = OptFom_Calculate_Settings(txffe, chdata, p, _op())
    assert len(S.H_r) == len(chdata[0].faxis)


def test_f_xc_length():
    """f_xc has N_fft_by2=512 points."""
    p = _param()
    txffe = np.array([[0.8]])
    S = OptFom_Calculate_Settings(txffe, _chdata(), p, _op())
    assert len(S.f_xc) == 512


def test_H_sy_ones_no_eta0():
    """H_sy = ones when USE_ETA0_PSD=False."""
    p = _param()
    op = _op()
    op.USE_ETA0_PSD = False
    txffe = np.array([[0.8]])
    S = OptFom_Calculate_Settings(txffe, _chdata(), p, op)
    assert np.allclose(S.H_sy, 1.0)


def test_phase_memory_shape():
    """phase_memory columns = number of txffe taps."""
    p = _param()
    txffe = np.ones((9, 3)) * 0.33
    S = OptFom_Calculate_Settings(txffe, _chdata(), p, _op())
    assert S.phase_memory.shape[1] == 3


def test_peak_search_range_within_data():
    """Peak_Search_Range values are valid 0-based indices."""
    p = _param()
    chdata = _chdata()
    txffe = np.array([[0.8]])
    S = OptFom_Calculate_Settings(txffe, chdata, p, _op())
    n = len(chdata[0].uneq_pulse_response)
    assert S.Peak_Search_Range[0] >= 0
    assert S.Peak_Search_Range[-1] < n


# ============================================================
# COM Octave oracle — the CL120d GDC-qual matrix, built by running
# OptFom_Calculate_Settings verbatim under Octave with
#   param.ctle_gdc_values = [-1 -3 -5 -7 -9 -11]
#   param.g_DC_HP_values  = [0 -1 -2 -3]
#   param.CTLE_type       = 'CL120d'
# and the gqual / g2qual below.
#
# Two things in this block do not survive a naive translation:
#
#   sort(g2qual,'descend') is STABLE, so tied values keep their input order.
#   np.argsort(a,'stable')[::-1] reverses them. Octave:
#       sort([3 1 3 2 1],'descend') -> index [1 3 4 2 5]
#   Reversing the stable ascending sort gives [3 1 4 5 2] instead, which
#   pairs each qual row with the wrong g2 window.
#
#   The gdc test reads gqual(kk,2) and gqual(kk,1) of the row after it is
#   sorted descending — the two LARGEST entries. min() and max() of the row
#   agree only while gqual has exactly two columns.
#
# And gqual(si,:) treats a bare [0 -6] as one row of two columns, where
# indexing a 1-D array picks out single elements.
# ============================================================
_OR_GDC = np.array([-1.0, -3, -5, -7, -9, -11])
_OR_GHP = np.array([0.0, -1, -2, -3])


def _qual_param(gqual, g2qual, M=8, fb=53.125e9):
    p = _param(M=M, fb=fb)
    p.CTLE_type = 'CL120d'
    p.ctle_gdc_values = _OR_GDC.copy()
    p.g_DC_HP_values = _OR_GHP.copy()
    p.gqual = np.asarray(gqual, dtype=float)
    p.g2qual = np.asarray(g2qual, dtype=float)
    return p


@pytest.mark.parametrize('gqual,g2qual,expected', [
    # tied g2qual: rows 2 and 3 of qual depend on the tie order
    ([[0, -6], [-6, -12], [-3, -9]], [0, -2, -2],
     [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1],
      [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]]),
    ([[-1, -5], [-2, -8], [-4, -10], [0, -6]], [-1, -1, -2, -1],
     [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]]),
    # three columns: the test uses the two largest, not min and max
    ([[0, -3, -6], [-6, -9, -12]], [0, -2],
     [[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0],
      [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]]),
    # one qual pair, written as a bare row
    ([0, -6], [0],
     [[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
    # no ties, for the ordinary case
    ([[0, -6], [-6, -12], [-3, -9]], [0, -2, -1],
     [[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0],
      [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]]),
])
def test_octave_oracle_cl120d_qual(gqual, g2qual, expected):
    S = OptFom_Calculate_Settings(np.zeros((1, 5)), _chdata(),
                                  _qual_param(gqual, g2qual), _op())
    np.testing.assert_array_equal(S.qual, np.asarray(expected, dtype=float))


def test_octave_oracle_single_column_gqual_refused():
    """Octave: gqual(_,2): out of bound 1 (dimensions are 2x1)."""
    with pytest.raises(IndexError):
        OptFom_Calculate_Settings(np.zeros((1, 5)), _chdata(),
                                  _qual_param([[0], [-6]], [0, -2]), _op())
