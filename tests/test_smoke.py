"""
Smoke tests: does sicopr.py import and do basic functions work?
Not checking numerical accuracy yet — just that nothing throws.
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sicopr


# ---------------------------------------------------------------------------
# Minimal synthetic helpers
# ---------------------------------------------------------------------------

def _param(fb=53.125e9, M=32, levels=4):
    p = SimpleNamespace()
    p.fb = fb
    p.samples_per_ui = M
    p.levels = levels
    p.ui = 1.0 / fb
    p.sample_dt = p.ui / M
    p.sigma_X = float(np.sqrt((levels**2 - 1) / (3 * (levels - 1)**2)))
    p.ndfe = 1
    p.specBER = 1e-4
    p.delta_y = 1e-4
    p.f_r = 0.75 * fb / 2
    p.BTorder = 4
    p.fb_BT_cutoff = 0.473037 * p.f_r
    p.fb_BW_cutoff = p.f_r
    p.ts_sample_adj_range = np.array([0.0, 0.0])
    p.Floating_DFE = False
    p.cursor_gain = np.array([0.0])
    p.CTLE_type = 'CL93'
    p.g_DC_HP_values = np.array([0.0])
    p.CTLE_fz  = np.array([3e9])
    p.CTLE_fp1 = np.array([9e9])
    p.CTLE_fp2 = np.array([18e9])
    p.ctle_gdc_values = np.array([0.0])
    p.snpPortsOrder = [1, 3, 2, 4]
    p.num_fext = 0
    p.num_next = 0
    p.num_s4p_files = 1
    p.sigma_ns = 0.0
    p.use_bmax = np.array([1.0])
    p.use_bmin = np.array([-1.0])
    p.bmax    = np.array([1.0])
    p.bmin    = np.array([-1.0])
    p.dfe_delta = 0.0
    p.N_bmax = 1
    p.R_LM = float((levels - 1) / levels)  # simplified
    p.SNR_TX = 35.0
    p.sigma_RJ = 0.001
    p.A_DD = 0.01
    p.eta_0 = 1e-3
    p.cursor_index = 1
    p.LOCAL_SEARCH = 0
    p.Floating_RXFFE = False
    p.GDC_MIN = 0
    p.base = 'smoke'
    p.tx_ffe_cm1_values = np.array([0.0])
    p.tx_ffe_cp1_values = np.array([0.0])
    p.tx_ffe_c0_min = -1.0
    return p


def _op():
    op = SimpleNamespace()
    op.DEBUG = False
    op.DISPLAY_WINDOW = False
    op.TDMODE = False
    op.GET_FD = False
    op.RxFFE = False
    op.RxFFE_with_MMSE = False
    op.FFE_OPT_METHOD = 'FOM'
    op.RX_CALIBRATION = False
    op.PSDRXCAL = False
    op.ERL_ONLY = False
    op.itick_box_size = 5
    op.Optimize_loop_speed_up = False
    op.TIME_AXIS = 'UI'
    op.BinSize = 1e-4
    op.impulse_response_truncation_threshold = 1e-4
    op.TS_SRCH_MODE = 'full-sweep'
    op.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN = False
    op.SNR_TXwC0 = False
    op.EXE_MODE = 0
    op.INCLUDE_CTLE = 0
    op.WC_PORTZ = False
    op.force_pdf_bin_size = False
    op.Histogram_Window_Weight = 'rectangle'
    op.RESULT_DIR = ''
    op.SNDR_REF = False
    return op


def _chdata(param):
    M = int(param.samples_per_ui)
    N = M * 64
    f_max = 2.0 * param.fb
    faxis = np.linspace(0, f_max, N // 2 + 1)
    cd = SimpleNamespace()
    cd.type = 'THRU'
    cd.filename = 'synthetic'
    cd.base = 'smoke'
    cd.faxis = faxis
    cd.sdd21 = np.ones(len(faxis), dtype=complex)
    cd.sdd21f = cd.sdd21.copy()
    cursor = N // 2
    pr = np.sinc((np.arange(N) - cursor) / M)
    cd.uneq_pulse_response = pr
    cd.eq_pulse_response = pr.copy()
    return [cd]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_import_ok():
    """sicopr.py must import without error."""
    import sicopr
    assert hasattr(sicopr, 'normal_dist')
    assert hasattr(sicopr, 'pdf_to_cdf')
    assert hasattr(sicopr, 'Bessel_Thomson_Filter')
    assert hasattr(sicopr, 'optimize_fom')


def test_bessel_smoke():
    """bessel(4) returns a 5-element array."""
    a = sicopr.bessel(4)
    assert len(a) == 5
    assert np.all(np.isfinite(a))


def test_bessel_thomson_filter_smoke():
    """Bessel_Thomson_Filter: DC gain = 1, all finite."""
    param = _param()
    faxis = np.linspace(0, 2 * param.fb, 200)
    H = sicopr.Bessel_Thomson_Filter(param, faxis, True)
    assert H.shape == faxis.shape
    assert np.all(np.isfinite(H))
    assert abs(H[0]) == pytest.approx(1.0, abs=1e-6)


def test_normal_dist_smoke():
    """normal_dist returns a valid PDF (sums to ~1)."""
    pdf = sicopr.normal_dist(0.01, 7, 1e-4)
    assert hasattr(pdf, 'x') and hasattr(pdf, 'y')
    assert abs(float(np.sum(pdf.y)) - 1.0) < 0.01


def test_pdf_to_cdf_smoke():
    """pdf_to_cdf returns a BER contour CDF.
    cdf.yB is the left-to-right cumsum and must end near 1.
    cdf.y is the BER contour (minimum of yB and yT) — NOT a 0→1 CDF.
    """
    pdf = sicopr.normal_dist(0.01, 8, 1e-4)
    cdf = sicopr.pdf_to_cdf(pdf)
    assert hasattr(cdf, 'yB') and hasattr(cdf, 'yT') and hasattr(cdf, 'y')
    assert float(cdf.yB[-1]) == pytest.approx(1.0, abs=1e-3), \
        f'yB[-1] should be 1.0, got {cdf.yB[-1]}'
    assert float(cdf.yB[0]) < 0.01


def test_optfom_build_txffe_smoke():
    """OptFom_Build_TXFFE returns 5 outputs with correct shapes."""
    param = _param()
    txffe_matrix, cur, sweep_idx, full_idx, cursor_vec = sicopr.OptFom_Build_TXFFE(param)
    assert isinstance(cur, (int, np.integer)), 'cur must be integer cursor position'
    assert txffe_matrix.ndim == 2, 'txffe_matrix must be 2D'
    assert txffe_matrix.shape[0] >= 1, 'must have at least one TXFFE combo'


def test_optfom_calc_fom_non_c2m():
    """OptFom_Calc_FOM(do_C2M=False) returns FOM = 20*log10(A_s/noise)."""
    THIS = SimpleNamespace(A_s=1.0, total_noise_rms=0.1, sigma_N=0.05, sigma_TX=0.03, cursor_i=40)
    param = SimpleNamespace(Noise_Crest_Factor=0.0, specBER=1e-4, Min_VEO_Test=0)
    op = SimpleNamespace(force_pdf_bin_size=False, BinSize=1e-4)
    FOM, skip = sicopr.OptFom_Calc_FOM(None, False, THIS, param, op, None)
    assert FOM == pytest.approx(20 * np.log10(1.0 / 0.1), abs=1e-6)
    assert skip == 0
