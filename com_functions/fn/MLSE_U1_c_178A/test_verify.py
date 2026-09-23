"""Tests for MLSE_U1_c_178A (MATLAB lines ~2348).

MATLAB GROUND TRUTH:
  A_s >> A_ni → COM_from_matlab large positive, DER_DFE < DER_CDR
  A_s << A_ni → DER_DFE > DER_CDR → delta_com=0, COM=COM_from_matlab
"""
import numpy as np
import pytest
from com_functions.fn.MLSE_U1_c_178A.py_impl import MLSE_U1_c_178A
from types import SimpleNamespace
import scipy.stats


def _make_gaussian_pdf(sigma, BinSize=1e-4):
    from types import SimpleNamespace as NS
    n_bins = int(6 * sigma / BinSize) * 2 + 1
    x_min = -(n_bins // 2)
    x = np.arange(x_min, -x_min + 1) * BinSize
    y = np.exp(-0.5 * (x / sigma) ** 2)
    y = y / np.sum(y)
    return NS(BinSize=BinSize, Min=int(x_min), y=y, x=x)


def _param(trunc=5):
    p = SimpleNamespace()
    p.num_ui_RXFF_noise = 20
    p.levels = 4
    p.fb = 26.5625e9
    p.specBER = 1e-4
    p.add_rx_noise = 0.0
    p.trunc = trunc
    p.DER_CDR = 1e-4
    p.Q_budget_adj = 0
    return p


def _psd_results(n_f=256):
    S = SimpleNamespace()
    S.S_n = np.ones(n_f) * 1e-6
    S.S_isi = np.zeros(n_f)
    S.S_rn = np.ones(n_f) * 1e-6
    S.S_rn_rms = 1e-3
    S.S_G_rms = 1e-3
    return S


def test_basic_high_snr():
    p = _param()
    sigma = 0.01
    pdf = _make_gaussian_pdf(sigma)
    CDF = np.cumsum(pdf.y)
    A_s = 1.0
    A_ni = 0.01
    b = np.array([0.0])
    r = MLSE_U1_c_178A(p, b, A_s, A_ni, pdf, CDF, _psd_results())
    assert r.COM_from_matlab > 30, "expected high COM"
    for f in ('COM', 'delta_com', 'DER_MLSE', 'DER_DFE', 'CDF', 'PDF'):
        assert hasattr(r, f)


def test_low_snr_path():
    p = _param()
    sigma = 0.5
    pdf = _make_gaussian_pdf(sigma)
    CDF = np.cumsum(pdf.y)
    A_s = 0.01
    A_ni = 1.0
    b = np.array([0.0])
    r = MLSE_U1_c_178A(p, b, A_s, A_ni, pdf, CDF, _psd_results())
    assert r.delta_com == 0.0
    assert r.COM == pytest.approx(r.COM_from_matlab, rel=1e-6)


def test_required_fields():
    p = _param()
    sigma = 0.05
    pdf = _make_gaussian_pdf(sigma)
    CDF = np.cumsum(pdf.y)
    r = MLSE_U1_c_178A(p, np.array([0.1]), 1.0, 0.05, pdf, CDF, _psd_results())
    for f in ('CDF', 'PDF', 'DER_MLSE_trunc', 'Q_budget_adj',
              'COM_from_matlab', 'DER_MLSE', 'DER_DFE',
              'COM', 'delta_com', 'delta_com_calc', 'g_an'):
        assert hasattr(r, f), f'missing field: {f}'


def test_com_nonnegative_delta():
    p = _param()
    sigma = 0.05
    pdf = _make_gaussian_pdf(sigma)
    CDF = np.cumsum(pdf.y)
    r = MLSE_U1_c_178A(p, np.array([0.0]), 1.0, 0.1, pdf, CDF, _psd_results())
    assert r.delta_com >= 0


def test_com_monotone_with_noise():
    p = _param()
    sigma1 = 0.01
    sigma2 = 0.05
    pdf1 = _make_gaussian_pdf(sigma1)
    pdf2 = _make_gaussian_pdf(sigma2)
    A_s = 1.0
    b = np.array([0.0])
    r1 = MLSE_U1_c_178A(p, b, A_s, 0.1, pdf1, np.cumsum(pdf1.y), _psd_results())
    r2 = MLSE_U1_c_178A(p, b, A_s, 0.1, pdf2, np.cumsum(pdf2.y), _psd_results())
    assert r1.COM >= r2.COM, "lower noise → higher COM"


# ---------------------------------------------------------------------------
# This module carries its own copy of scalePDF as the private _scale_pdf.
# interp1's default returns NaN OUTSIDE the data range; np.interp clamps, which
# makes the reference's two "NAN interp work around" lines (patching only y(1)
# and y(end)) no-ops. That is harmless only while at most one point falls
# outside at each end.
#
# COM Octave on the canonical scalePDF, Min=-8, x=(-8:0)*0.05, scale 1.0:
# 17 values, ALL NaN. Clamping gave 17 finite ones. Pinned here so the copy
# cannot drift back -- a canonical fix does not reach an inlined copy, which is
# the defect class that cost engine defect #6.
# ---------------------------------------------------------------------------

def test_scale_pdf_copy_returns_nan_outside_the_data_range():
    from com_functions.fn.MLSE_U1_c_178A.py_impl import _scale_pdf
    y = np.array([0.02, 0.05, 0.09, 0.14, 0.20, 0.22, 0.15, 0.08, 0.05])
    x = np.arange(-8, 1) * 0.05          # left-heavy: max(x)=0 < -min(x)=0.4
    out = _scale_pdf(SimpleNamespace(BinSize=0.05, Min=-8, x=x, y=y), 1.0)
    got = np.asarray(out.y)
    assert got.size == 17
    assert np.isnan(got).all()


def test_scale_pdf_copy_keeps_an_ordinary_pdf_finite():
    """The guard: a symmetric grid must not become NaN."""
    from com_functions.fn.MLSE_U1_c_178A.py_impl import _scale_pdf
    y = np.array([0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    x = np.arange(-3, 3) * 0.05
    got = np.asarray(_scale_pdf(SimpleNamespace(BinSize=0.05, Min=-3, x=x, y=y), 1.0).y)
    assert np.isfinite(got).all()
    assert abs(float(np.sum(got)) - 1.0) < 1e-12
