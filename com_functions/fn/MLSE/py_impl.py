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
from scipy.special import erfc, erfcinv
from types import SimpleNamespace


def _qfuncinv(x):
    return float(np.sqrt(2) * erfcinv(2 * x))


def _qfunc(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2))


def _CDF_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    cdf = np.asarray(CDF, dtype=float)
    index = int(np.argmax(x >= -val))
    return float(cdf[index])


def _CDF_inv_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    cdf = np.asarray(CDF, dtype=float)
    indices = np.where(cdf >= val)[0]
    if len(indices) == 0:
        return float(x[-1])
    return float(x[indices[0]])


def MLSE(param, alpha, A_s, A_ni, PDF, CDF):
    """MLSE analysis (MATLAB lines 2266-2347).

    Returns MLSE_results SimpleNamespace.
    """
    pdf_x = np.asarray(PDF.x, dtype=float)
    pdf_y = np.asarray(PDF.y, dtype=float)
    cdf = np.asarray(CDF, dtype=float)

    COM_from_matlab = 20.0 * np.log10(A_s / A_ni)
    L = int(param.levels)
    k_DER = _qfuncinv(param.specBER)
    sigma_noise = float(np.sqrt(np.sum(pdf_y * pdf_x ** 2)))
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
            DER_delta = 1.0 - last / DER_MLSE_CDF if DER_MLSE_CDF != 0 else np.inf
            jj += 1

        inner_g = (0.5 * DER_MLSE * (L / (L - 1) - _qfunc((1 - 2 * alpha) * A_peak / ((L - 1) * sigma_noise))))
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
