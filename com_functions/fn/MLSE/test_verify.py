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


# ============================================================
# COM Octave oracle — MLSE run verbatim under Octave.
#
# CDF_ev was supplied from matlab/com_ieee8023_4p16p0.m, not from the compat
# file: the compat build replaces its find() with a lookup() speed variant and
# is not the reference for this call.
#
# PDF is the Gaussian of _gaussian_pdf (sigma 0.05, bin 0.001), CDF its cumsum,
# param.levels = 4, param.specBER = 1e-6, A_s = 0.5, A_ni = 0.1:
#
#   alpha = 0     SNR_DFE                    500.00708233972136
#                 sigma_noise                  0.049999645886775849
#                 SNR_dB                      26.989761559345709
#                 DER_MLSE_CDF                 7.1128462102046472e-07
#                 SNR_DFE_eqivalent_CDF      116.16364539749341
#                 COM_CDF                      7.6403408514973679
#                 delta_com_CDF               -6.3390592352230088
#                 SNR_DFE_eqivalent_Gaussian 504.54385532432599
#                 delta_com_Gaussian           0.039227654759820568
#   alpha = 0.3   SNR_DFE_eqivalent_Gaussian 549.61861284842109
#                 delta_com_Gaussian           0.41085275759730411
#
# The Gaussian pair is what pins the argument of qfunc in step 4. MATLAB
# writes `qfunc((1-2*alpha)*main/(L-1)*sigma_noise)` — main/(L-1) MULTIPLIED
# by sigma_noise. Dividing by it instead (which the commented-out DER_DFE line
# above does) gives 500.007 and a delta of 0 at alpha = 0.
#
# They are pinned to 1e-8 relative rather than to the bit because both go
# through qfuncinv, and Octave's erfcinv is the weak link: for specBER 1e-6 it
# answers 4.7534243088333916 where scipy answers 4.7534243088228987, and
# feeding each back through Q gives a relative error of 5.2e-11 for Octave
# against 3.8e-15 for scipy. k_DER is therefore NOT pinned to Octave.
# ============================================================
def _oracle_inputs(levels=4, ber=1e-6):
    pdf = _gaussian_pdf()
    return _param(levels=levels, specBER=ber), pdf, np.cumsum(pdf.y)


def test_octave_oracle_cdf_branch():
    p, pdf, cdf = _oracle_inputs()
    r = MLSE(p, 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    assert r.SNR_DFE == pytest.approx(500.00708233972136, rel=1e-12)
    assert r.sigma_noise == pytest.approx(0.049999645886775849, rel=1e-12)
    assert r.SNR_dB == pytest.approx(26.989761559345709, rel=1e-12)
    assert r.DER_MLSE_CDF == pytest.approx(7.1128462102046472e-07, rel=1e-12)
    assert r.SNR_DFE_eqivalent_CDF == pytest.approx(116.16364539749341, rel=1e-12)
    assert r.COM_CDF == pytest.approx(7.6403408514973679, rel=1e-12)
    assert r.delta_com_CDF == pytest.approx(-6.3390592352230088, rel=1e-12)


@pytest.mark.parametrize('levels,alpha,A_s,A_ni,snr_eq,dcom', [
    (4, 0.0, 0.5, 0.1, 504.54385532432599, 0.039227654759820568),
    (4, 0.3, 0.5, 0.1, 549.61861284842109, 0.41085275759730411),
    (4, 0.5, 0.6, 0.15, 904.68716283487947, 0.99159779857862573),
    (2, 0.2, 0.4, 0.08, 67.121777642745386, 0.2067732417819213),
])
def test_octave_oracle_gaussian_step4_multiplies_by_sigma(levels, alpha, A_s, A_ni,
                                                          snr_eq, dcom):
    p, pdf, cdf = _oracle_inputs(levels=levels, ber=1e-6 if levels == 4 else 1e-5)
    r = MLSE(p, alpha, A_s=A_s, A_ni=A_ni, PDF=pdf, CDF=cdf)
    assert r.SNR_DFE_eqivalent_Gaussian == pytest.approx(snr_eq, rel=1e-8)
    # absolute, not relative: delta_com is 10*log10 of a ratio near 1, so the
    # qfuncinv difference above is amplified by the cancellation.
    assert r.delta_com_Gaussian == pytest.approx(dcom, abs=1e-7)


# ============================================================
# COM Octave oracle — param.levels is a double, not an integer.
# With param.levels = 4.5 and everything else as above, Octave gives
#   SNR_DFE 641.67575566930918   SNR_dB 28.073156307234093
#   DER_MLSE_CDF 9.3357812919632214e-07   COM_CDF 7.5679580189627558
# int(param.levels) truncated to 4 and returned COM_CDF 7.6403408514973679.
# ============================================================
def test_octave_oracle_fractional_levels():
    p, pdf, cdf = _oracle_inputs(levels=4.5)
    r = MLSE(p, 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=cdf)
    assert r.SNR_DFE == pytest.approx(641.67575566930918, rel=1e-12)
    assert r.SNR_dB == pytest.approx(28.073156307234093, rel=1e-12)
    assert r.DER_MLSE_CDF == pytest.approx(9.3357812919632214e-07, rel=1e-12)
    assert r.COM_CDF == pytest.approx(7.5679580189627558, rel=1e-12)


# ============================================================
# COM Octave oracle — the two places MATLAB divides by zero and carries on.
#
# All the PDF mass in one bin, so sigma_noise is 0:
#   sigma_noise 0   SNR_dB Inf   DER_MLSE_CDF 0
#   SNR_DFE_eqivalent_CDF Inf   COM_CDF NaN   delta_com_CDF NaN
# The port used Python floats there and raised ZeroDivisionError.
#
# A CDF whose first entry is exactly 0, so CDF_ev answers 0 for every jj:
#   DER_MLSE_CDF 0   SNR_DFE_eqivalent_CDF 180.00001316021442
#   COM_CDF 9.5424250943932485   SNR_dB 26.989700360882871
# MATLAB leaves the while loop on the first pass, because
# DER_delta = 1-0/0 is NaN and `while NaN > .001` is false. The port special-
# cased that to inf and never left the loop, so the call never returned.
# ============================================================
def test_octave_oracle_zero_sigma_noise_is_infinite_not_an_error():
    x = np.arange(-300, 301) * 0.001
    y = np.zeros_like(x)
    y[300] = 1.0
    pdf = SimpleNamespace(x=x, y=y)
    r = MLSE(_param(), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=np.cumsum(y))
    assert r.sigma_noise == 0.0
    assert r.SNR_dB == np.inf
    assert r.DER_MLSE_CDF == 0.0
    assert r.SNR_DFE_eqivalent_CDF == np.inf
    assert np.isnan(r.COM_CDF)
    assert np.isnan(r.delta_com_CDF)


def test_octave_oracle_series_stuck_at_zero_terminates(monkeypatch):
    """The while loop must stop, so the call is driven through a CDF_ev that
    refuses to be called more than a few times. On the old body it never
    stopped, and the test would otherwise hang rather than fail."""
    import com_functions.fn.MLSE.py_impl as mod

    real = mod._CDF_ev
    calls = {'n': 0}

    def counted(val, PDF, CDF):
        calls['n'] += 1
        if calls['n'] > 50:
            raise RuntimeError('CDF_ev called %d times: the DER_MLSE_CDF loop '
                               'is not terminating' % calls['n'])
        return real(val, PDF, CDF)

    monkeypatch.setattr(mod, '_CDF_ev', counted)

    sigma = 0.05
    x = np.arange(-300, 301) * 0.001
    y = np.exp(-x ** 2 / (2 * sigma ** 2))
    y /= y.sum()
    y[0] = 0.0                      # CDF[0] == 0, so CDF_ev answers 0 for all jj
    pdf = SimpleNamespace(x=x, y=y)
    r = MLSE(_param(), 0.0, A_s=0.5, A_ni=0.1, PDF=pdf, CDF=np.cumsum(y))
    assert r.DER_MLSE_CDF == 0.0
    assert r.SNR_dB == pytest.approx(26.989700360882871, rel=1e-12)
    assert r.SNR_DFE_eqivalent_CDF == pytest.approx(180.00001316021442, rel=1e-12)
    assert r.COM_CDF == pytest.approx(9.5424250943932485, rel=1e-12)
