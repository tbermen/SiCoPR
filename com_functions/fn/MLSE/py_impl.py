# ============================================================
# MATLAB→Python translation notes for MLSE
# MATLAB lines: 2266–2347
# ============================================================
# qfuncinv(x) = sqrt(2)*erfcinv(2*x) → scipy.special.erfcinv.
# qfunc(x) = 0.5*erfc(x/sqrt(2)) → scipy.special.erfc.
# CDF_ev, CDF_inv_ev: inlined as _CDF_ev and _CDF_inv_ev.
# DER_MLSE loop: j = 1:200 (Python range(1,201)).
# DER_MLSE_CDF while loop: convergence when DER_delta <= 0.001.
# When A_s < A_ni: skip computation, set new_com_CDF = COM_from_matlab, deltas=0.
# Fields populated: COM_from_matlab, SNR_DFE, DER_MLSE_Gaussian, DER_MLSE_CDF,
#   sigma_noise, SNR_dB, SNR_DFE_eqivalent_Gaussian, SNR_DFE_eqivalent_CDF,
#   COM_Gaussian, COM_CDF, k_DER, delta_com_CDF, delta_com_Gaussian.
# ============================================================

import numpy as np
from com_functions.fn.CDF_ev.py_impl import CDF_ev as _CDF_ev
from com_functions.fn.CDF_inv_ev.py_impl import CDF_inv_ev as _CDF_inv_ev
from scipy.special import erfc, erfcinv
from types import SimpleNamespace


def _qfuncinv(x):
    return float(np.sqrt(2) * erfcinv(2 * x))


def _qfunc(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def MLSE(param, alpha, A_s, A_ni, PDF, CDF):
    """MLSE analysis (MATLAB lines 2266-2347).

    Returns MLSE_results SimpleNamespace.
    """
    pdf_x = np.asarray(PDF.x, dtype=float)
    pdf_y = np.asarray(PDF.y, dtype=float)
    cdf = np.asarray(CDF, dtype=float)

    # np.float64 rather than Python floats: MATLAB divides by zero without
    # complaining and lets the Inf or NaN flow out, where a Python float
    # raises ZeroDivisionError. COM Octave, with all the PDF mass at x=0 so
    # sigma_noise is 0: SNR_dB Inf, SNR_DFE_eqivalent_CDF Inf, COM_CDF NaN.
    # The port raised instead, on a PDF the reference answers for. errstate
    # keeps the Inf and NaN quiet, as MATLAB produces them without a warning.
    with np.errstate(divide='ignore', invalid='ignore'):
        A_s = np.float64(A_s)
        A_ni = np.float64(A_ni)
        COM_from_matlab = 20.0 * np.log10(A_s / A_ni)
        # L, not int(L). MATLAB keeps param.levels as the double it is; int()
        # turned a levels of 4.5 into 4 and moved every result (COM_CDF 7.5680
        # in COM Octave against 7.6403 here).
        L = np.float64(param.levels)
        k_DER = _qfuncinv(param.specBER)
        sigma_noise = np.sqrt(np.sum(pdf_y * pdf_x ** 2))
        A_peak = (L - 1) * A_s
        SNR_dB = float(10.0 * np.log10(
            1.0 / 3.0 * (L + 1) / (L - 1) * A_peak ** 2 / sigma_noise ** 2))

        if A_s >= A_ni:
            SNR_DFE = float(1.0 / 3.0 * (L + 1) / (L - 1) * A_peak ** 2 / sigma_noise ** 2)

            # DER_MLSE (Gaussian): vectorised sum over j=1..200
            j = np.arange(1, 201, dtype=float)
            arg = np.sqrt(1 + (j - 1) * (1 - alpha) ** 2 + alpha ** 2) * A_peak / ((L - 1) * sigma_noise)
            DER_MLSE = float(2.0 * np.sum(j * ((L - 1) / L) ** j * _qfunc(arg)))

            # DER_MLSE_CDF: converging series
            DER_MLSE_CDF = 0.0
            jj = 1
            DER_delta = np.inf
            while DER_delta > 0.001:
                last = DER_MLSE_CDF
                arg_jj = float(np.sqrt(1 + (jj - 1) * (1 - alpha) ** 2 + alpha ** 2) * A_peak / (L - 1))
                term = 2.0 * jj * ((L - 1) / L) ** jj * _CDF_ev(arg_jj, PDF, cdf)
                DER_MLSE_CDF = term + DER_MLSE_CDF
                # MATLAB: DER_delta = 1-last_DER_MLSE_CDF/DER_MLSE_CDF, with IEEE
                # division. A series stuck at zero gives 1-0/0 = NaN, and
                # `while NaN > .001` is false, so MATLAB leaves the loop and
                # reports DER_MLSE_CDF 0. Substituting inf for that case never
                # left the loop at all. COM Octave, on a PDF whose CDF starts at
                # exactly 0: DER_MLSE_CDF 0, SNR_DFE_eqivalent_CDF Inf,
                # COM_CDF NaN.
                DER_delta = 1.0 - np.float64(last) / np.float64(DER_MLSE_CDF)
                jj += 1

            # MATLAB writes `main/(L-1)*sigma_noise` here, a MULTIPLY: the
            # expression is ((1-2*alpha)*main/(L-1)) * sigma_noise, not
            # (1-2*alpha)*main/((L-1)*sigma_noise). COM Octave, PAM4 alpha=0,
            # A_s=0.5, A_ni=0.1, sigma_noise 0.05:
            #     SNR_DFE_eqivalent_Gaussian 504.54385532432599  (dividing: 500.007)
            #     delta_com_Gaussian           0.039227654759821 (dividing: 0)
            # and at alpha=0.3, 549.61861284842109 against 545.00795513714922.
            # The commented-out DER_DFE line just above it in the reference does
            # divide; this line does not, and the port followed the comment.
            inner_g = (0.5 * DER_MLSE * (L / (L - 1) - _qfunc((1 - 2 * alpha) * A_peak / (L - 1) * sigma_noise)))
            SNR_DFE_eqivalent = float(SNR_DFE * ((L - 1) * sigma_noise / A_peak * _qfuncinv(float(inner_g))) ** 2)

            inner_cdf = 0.5 * DER_MLSE_CDF * (L / (L - 1) - _CDF_ev(float((1 - 2 * alpha) * A_peak / (L - 1)), PDF, cdf))
            SNR_DFE_eqivalent_CDF = float(SNR_DFE * ((L - 1) / A_peak * _CDF_inv_ev(inner_cdf, PDF, cdf)) ** 2)

            delta_com = float(10.0 * np.log10(SNR_DFE_eqivalent / SNR_DFE))
            delta_com_CDF = float(10.0 * np.log10(SNR_DFE_eqivalent_CDF / SNR_DFE))
            new_com_CDF = float(COM_from_matlab + delta_com_CDF)
        else:
            SNR_DFE = None
            DER_MLSE = None
            DER_MLSE_CDF = None
            SNR_DFE_eqivalent = None
            SNR_DFE_eqivalent_CDF = None
            new_com_CDF = float(COM_from_matlab)
            delta_com_CDF = 0.0
            delta_com = 0.0

    r = SimpleNamespace()
    r.COM_from_matlab = float(COM_from_matlab)
    r.SNR_DFE = SNR_DFE
    r.DER_MLSE_Gaussian = DER_MLSE
    r.DER_MLSE_CDF = DER_MLSE_CDF
    r.sigma_noise = float(sigma_noise)
    r.SNR_dB = float(SNR_dB)
    r.SNR_DFE_eqivalent_Gaussian = SNR_DFE_eqivalent
    r.SNR_DFE_eqivalent_CDF = SNR_DFE_eqivalent_CDF
    r.COM_Gaussian = float(new_com_CDF)
    r.COM_CDF = float(new_com_CDF)
    r.k_DER = float(k_DER)
    r.delta_com_CDF = float(delta_com_CDF)
    r.delta_com_Gaussian = float(delta_com)
    return r
