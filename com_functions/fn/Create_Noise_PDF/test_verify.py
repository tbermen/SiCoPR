"""Verification tests for Create_Noise_PDF().

# ============================================================
# MATLAB GROUND TRUTH (lines 1552-1680)
# Computes combined interference and noise PDF/CDF and NS noise struct.
# Non-MMSE, RX_CALIBRATION=0: uses sigma_TX from SNR_TX param.
# N_qb=0: skips quantization step.
# Returns (PDF, CDF, NS).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com_functions.fn.Create_Noise_PDF.py_impl as _mod
from com_functions.fn.Create_Noise_PDF.py_impl import Create_Noise_PDF


def _gaussian_pdf(sigma=0.05, bin_size=0.001):
    x = np.arange(-round(5*sigma/bin_size), round(5*sigma/bin_size)+1) * bin_size
    y = np.exp(-x**2/(2*sigma**2)); y /= y.sum()
    return SimpleNamespace(BinSize=bin_size, Min=int(round(x[0]/bin_size)), y=y, x=x)


def _delta_pdf(bin_size=0.001):
    return SimpleNamespace(BinSize=bin_size, Min=0, y=np.array([1.0]), x=np.array([0.0]))


def _param():
    bs = 0.001
    return SimpleNamespace(
        levels=4, specBER=1e-6,
        R_LM=0.4, SNR_TX=30.0,
        sigma_RJ=0.01, sigma_X=1.0,
        A_DD=0.01,
        delta_y=bs,
        N_qb=0,
        Noise_Crest_Factor=0,
        CTLE_type='CL93',
        ctle_gdc_values=np.array([-6.0, 0.0]),
        CTLE_fp1=np.array([10e9]), CTLE_fp2=np.array([20e9]),
        CTLE_fz=np.array([5e9]),
        g_DC_HP_values=np.array([0.0]),
        f_HP=np.array([1e9]),
        f_HP_Z=np.array([1e9]), f_HP_P=np.array([2e9]),
        number_of_s4p_files=1,
    )


def _fom_result():
    return SimpleNamespace(
        sigma_N=0.02,
        ctle=1,
        best_G_high_pass=1,
        txffe=np.array([0.0, 1.0, 0.0]),
        cur=2,
        h_J=np.zeros(5),
    )


def _OP():
    return SimpleNamespace(
        RX_CALIBRATION=0,
        FFE_OPT_METHOD='',
        RxFFE=False,
        SNR_TXwC0=False,
        PSDRXCAL=False,
        force_BBN_Q_factor=False,
        BBN_Q_factor=7.0,
    )


def _chdata():
    bs = 0.001
    pdfr = _delta_pdf(bs)
    return [SimpleNamespace(type='THRU', pdfr=pdfr)]


def test_returns_three_outputs():
    """Create_Noise_PDF returns (PDF, CDF, NS)."""
    PDF, CDF, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert hasattr(PDF, 'y')
    assert len(CDF) > 0
    assert hasattr(NS, 'sigma_N')


def test_pdf_y_sums_to_one():
    """Combined PDF.y sums to 1."""
    PDF, CDF, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert float(np.sum(PDF.y)) == pytest.approx(1.0, abs=1e-4)


def test_cdf_monotone():
    """CDF is non-decreasing."""
    _, CDF, _ = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert np.all(np.diff(CDF) >= -1e-12)


def test_sigma_TX_positive():
    """NS.sigma_TX > 0."""
    _, _, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert NS.sigma_TX > 0


def test_ns_has_required_fields():
    """NS has sigma_N, sigma_TX, sigma_G, ber_q, noise_pdf."""
    _, _, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    for field in ['sigma_N', 'sigma_TX', 'sigma_G', 'ber_q', 'noise_pdf',
                  'sci_pdf', 'gaussian_noise_pdf', 'p_DD', 'jitt_pdf']:
        assert hasattr(NS, field), f'Missing: {field}'


def test_N_qb_nonzero_invokes_quantization(monkeypatch):
    """N_qb != 0 routes through adjust_Rx_noise_for_quantization (no longer a stub).

    The quantization helper is a top-level fn in the assembled module; here we
    inject a spy to confirm the dispatch reaches it and the combined PDF/CDF are
    still returned. (The helper itself is unit-tested in its own directory.)"""
    p = _param()
    p.N_qb = 4
    calls = {'n': 0}

    def spy_adjust(pdf, NS, chdata, fom_result, param, OP):
        calls['n'] += 1
        return chdata, NS, pdf  # pass the combined PDF through unchanged

    monkeypatch.setattr(_mod, 'adjust_Rx_noise_for_quantization', spy_adjust, raising=False)
    PDF, CDF, NS = Create_Noise_PDF(0.5, p, _fom_result(), _chdata(), _OP(), 0.0)
    assert calls['n'] == 1
    assert hasattr(PDF, 'y')
    assert len(CDF) == len(PDF.y)
