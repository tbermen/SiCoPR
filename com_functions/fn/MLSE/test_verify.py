"""Verification tests for MLSE().

# ============================================================
# MATLAB GROUND TRUTH (lines 2266-2347)
# MLSE analysis adjusting COM by DER-MLSE factor.
# A_s >= A_ni path: computes DER_MLSE, DER_MLSE_CDF, SNR_DFE_eqivalent.
# A_s < A_ni path: warning, new_com_CDF = COM_from_matlab, deltas=0.
# Returns MLSE_results with COM_Gaussian, COM_CDF, delta_com_CDF, etc.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
from scipy.special import erfcinv
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.MLSE.py_impl import MLSE


def _gaussian_pdf(sigma=0.05, bin_size=0.001):
    x = np.arange(-round(5*sigma/bin_size), round(5*sigma/bin_size)+1) * bin_size
    y = np.exp(-x**2/(2*sigma**2)); y /= y.sum()
    return SimpleNamespace(x=x, y=y)


def _param(levels=4, specBER=1e-6):
    return SimpleNamespace(levels=levels, specBER=specBER)


def test_returns_struct():
    """MLSE returns a struct with COM_CDF."""
    pdf = _gaussian_pdf()
    cdf = np.cumsum(pdf.y)
    r = MLSE(_param(), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    assert hasattr(r, 'COM_CDF')


def test_com_from_matlab_matches():
    """COM_from_matlab = 20*log10(A_s/A_ni)."""
    pdf = _gaussian_pdf()
    cdf = np.cumsum(pdf.y)
    r = MLSE(_param(), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    assert r.COM_from_matlab == pytest.approx(20*np.log10(0.5/0.1), rel=1e-4)


def test_low_signal_path():
    """A_s < A_ni: delta_com_CDF=0, COM_CDF=COM_from_matlab."""
    pdf = _gaussian_pdf()
    cdf = np.cumsum(pdf.y)
    r = MLSE(_param(), 0.0, A_s=0.05, A_ni=0.5, PDF=pdf, CDF=cdf)
    assert r.delta_com_CDF == pytest.approx(0.0)
    assert r.COM_CDF == pytest.approx(r.COM_from_matlab)


def test_sigma_noise_positive():
    """sigma_noise > 0 for non-trivial PDF."""
    pdf = _gaussian_pdf(sigma=0.05)
    cdf = np.cumsum(pdf.y)
    r = MLSE(_param(), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    assert r.sigma_noise > 0


def test_k_DER_correct():
    """k_DER = sqrt(2)*erfcinv(2*specBER)."""
    pdf = _gaussian_pdf()
    cdf = np.cumsum(pdf.y)
    r = MLSE(_param(specBER=1e-6), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    expected = float(np.sqrt(2) * erfcinv(2 * 1e-6))
    assert r.k_DER == pytest.approx(expected, rel=1e-4)


def test_all_fields_present():
    """MLSE_results has all expected fields."""
    pdf = _gaussian_pdf()
    cdf = np.cumsum(pdf.y)
    r = MLSE(_param(), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    for field in ['COM_from_matlab', 'SNR_DFE', 'DER_MLSE_Gaussian', 'DER_MLSE_CDF',
                  'sigma_noise', 'SNR_dB', 'SNR_DFE_eqivalent_Gaussian',
                  'SNR_DFE_eqivalent_CDF', 'COM_Gaussian', 'COM_CDF',
                  'k_DER', 'delta_com_CDF', 'delta_com_Gaussian']:
        assert hasattr(r, field), f'Missing field: {field}'
