# ============================================================
# MATLAB→Python translation notes for MLSE_U1_c_178A
# MATLAB lines: 2348–2477
# ============================================================
# Depends on: scaleCDF, CDF_ev, CDF_inv_ev, scalePDF, conv_fct (all implemented — inlined).
# S_ni = S_isi + S_n + S_an: PSD sum (arrays).
# R_ni = ifft(S_ni) * fb: correlation vector.
# p_j = conv_fct(p_an, p_scaled_by_b): start, then convolve p_scaled_by_1mb each iter.
# rou = R_ni / R_ni[0] (normalised).
# toeplitz(rou[0:j+1]) for V_j at each j.
# u_j: alternating sign vector of length j+2.
# DER_MLSE_trunc: accumulates separately up to param.trunc.
# delta_com: computed if DER_DFE <= DER_CDR; else 0.
# Q_budget_adj: 0 or linear fn of COM_from_matlab.
# Truncation warning: print instead of msgbox.
# ============================================================

import copy
import numpy as np
from com_functions.fn.CDF_ev.py_impl import CDF_ev as _CDF_ev
from com_functions.fn.CDF_inv_ev.py_impl import CDF_inv_ev as _CDF_inv_ev
from com_functions.fn.scaleCDF.py_impl import scaleCDF as _scaleCDF
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
from scipy.special import erfc
from types import SimpleNamespace



# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


def _scale_pdf(pdf, scale_factor):
    if scale_factor == 0.0:
        pdf_out = copy.copy(pdf)
        pdf_out.Min = 0
        pdf_out.x = np.array([0.0])
        pdf_out.y = np.array([1.0])
        return pdf_out
    pdf_out = copy.copy(pdf)
    pdf_out.Min = int(np.floor(pdf.Min * scale_factor))
    idx = np.arange(pdf_out.Min, -pdf_out.Min + 1)
    pdf_out.x = idx * pdf_out.BinSize
    pdf_out.y = np.interp(pdf_out.x, np.asarray(pdf.x) * scale_factor, np.asarray(pdf.y))
    if len(pdf_out.y) > 1:
        pdf_out.y[0] = pdf_out.y[1]
        pdf_out.y[-1] = pdf_out.y[-2]
    pdf_out.y = pdf_out.y / np.sum(pdf_out.y)
    return pdf_out


def MLSE_U1_c_178A(param, b, A_s, A_ni, PDF, CDF, PSD_results):
    """MLSE analysis per IEEE 178A (MATLAB lines 2348-2477).

    Returns MLSE_results SimpleNamespace.
    b: DFE tap vector (scalar or array; b[0] used).
    """
    num_ui = int(param.num_ui_RXFF_noise)
    L = int(param.levels)
    fb = float(param.fb)
    DER0 = float(param.specBER)
    delta_COM_an = float(param.add_rx_noise)

    b_arr = np.asarray(b, dtype=float).ravel()
    b0 = float(b_arr[0]) if len(b_arr) > 0 else 0.0

    pdf_x = np.asarray(PDF.x, dtype=float)
    pdf_y = np.asarray(PDF.y, dtype=float)
    cdf_arr = np.asarray(CDF, dtype=float)

    # Step 1: scale CDF
    p_an, P_an, _ = _scaleCDF(PDF, delta_COM_an, DER0, A_s)

    sigma_an_2_pdf = float(np.sum(p_an.y * p_an.x ** 2) - np.sum(pdf_y * pdf_x ** 2))
    sigma_G_2 = float(PSD_results.S_G_rms) ** 2
    g_an = sigma_an_2_pdf / (float(PSD_results.S_rn_rms) ** 2)

    COM_from_matlab = float(20.0 * np.log10(A_s / A_ni))
    DER_DFE = _CDF_ev(A_s, PDF, cdf_arr)

    S_an = g_an * np.asarray(PSD_results.S_rn, dtype=float)
    S_ni = np.asarray(PSD_results.S_isi, dtype=float) + np.asarray(PSD_results.S_n, dtype=float) + S_an
    R_ni = np.real(np.fft.ifft(S_ni)) * fb
    rou = R_ni / (R_ni[0] if R_ni[0] != 0 else 1.0)

    # Build convolved PDFs
    p_scaled_by_b = _scale_pdf(p_an, b0)
    p_j = _conv_fct(p_an, p_scaled_by_b)
    p_scaled_by_1mb = _scale_pdf(p_an, 1.0 - b0)
    p_trunc = copy.copy(p_an)

    trunc = int(getattr(param, 'trunc', num_ui // 2))
    DER_CDR = float(getattr(param, 'DER_CDR', DER0))

    if DER_DFE <= DER_CDR:
        j = 1
        DER_MLSE = 0.0
        DER_MLSE_j = np.inf
        DER_MLSE_trunc = 0.0
        smallest_relative_change = 0.0001

        while j <= num_ui // 2 and DER_MLSE_j > DER_MLSE * smallest_relative_change:
            u_j = np.zeros(j + 2)
            u_j[0] = 1.0
            u_j[1:j] = (1.0 - b0)
            u_j[j] = (-1.0) ** (j + 1) * b0
            for k in range(1, j, 2):
                u_j[k] = -u_j[k]

            n_v = min(j + 1, len(rou))
            V_j = toeplitz(rou[:n_v])
            uj_sub = u_j[:n_v]
            uu = float(uj_sub @ uj_sub)
            uVu = float(uj_sub @ V_j @ uj_sub)
            scale_arg = float(A_s * uu ** 1.5 / (uVu ** 0.5)) if uVu > 0 else 0.0
            P_j_y = np.cumsum(p_j.y)
            DER_MLSE_j = float(((L - 1) / L) ** (j - 1)
                               * _CDF_ev(scale_arg, p_j, P_j_y))
            DER_MLSE += DER_MLSE_j

            if j == trunc:
                V_trunc = toeplitz(rou[:j])
                u_trunc = u_j[:j]
                uu_t = float(u_trunc @ u_trunc)
                uVu_t = float(u_trunc @ V_trunc @ u_trunc)
                scale_t = float(A_s * uu_t ** 1.5 / (uVu_t ** 0.5)) if uVu_t > 0 else 0.0
                P_trunc_y = np.cumsum(p_trunc.y)
                DER_MLSE_trunc += (L * ((L - 1) / L) ** (j - 1)
                                   * _CDF_ev(scale_t, p_trunc, P_trunc_y))
            elif j < trunc:
                DER_MLSE_trunc = DER_MLSE
                p_trunc = _conv_fct(p_trunc, p_scaled_by_1mb)

            p_j = _conv_fct(p_j, p_scaled_by_1mb)
            j += 1

        # Q_budget_adj
        Q_budget_adj_param = getattr(param, 'Q_budget_adj', 0)
        if np.ndim(Q_budget_adj_param) == 0 and float(Q_budget_adj_param) == 0:
            Q_budget_adj = 0.0
        else:
            q_arr = np.asarray(Q_budget_adj_param, dtype=float).ravel()
            Q_budget_adj = float(q_arr[0] - q_arr[1] * COM_from_matlab) if len(q_arr) >= 2 else 0.0

        inv_trunc = _CDF_inv_ev(DER_MLSE_trunc, p_an, P_an)
        delta_com = float(20.0 * np.log10(1.0 / A_s * -inv_trunc)) - Q_budget_adj if inv_trunc < 0 else 0.0
        inv_notrunc = _CDF_inv_ev(DER_MLSE, p_an, P_an)
        delta_com_notrunc = (float(20.0 * np.log10(1.0 / A_s * -inv_notrunc)) - Q_budget_adj
                             if inv_notrunc < 0 else 0.0)
        delta_com_notrunc = max(delta_com_notrunc, 0.0)
        DER_MLSE = _CDF_ev(A_s * 10.0 ** (delta_com_notrunc / 20.0), PDF, cdf_arr)
        delta_com_calc = delta_com
        if delta_com < 0:
            delta_com = 0.0
            print('MLSE truncation failed. Try increasing trunc')
        DER_MLSE_trunc = _CDF_ev(A_s * 10.0 ** (delta_com / 20.0), PDF, cdf_arr)
        new_com = COM_from_matlab + delta_com
    else:
        DER_MLSE = float('nan')
        new_com = COM_from_matlab
        delta_com = 0.0
        delta_com_calc = 0.0
        Q_budget_adj = 0.0
        DER_MLSE_trunc = float('nan')

    PDF1, CDF1, _ = _scaleCDF(PDF, -delta_com, DER0, A_s)

    r = SimpleNamespace()
    r.CDF = CDF1
    r.PDF = PDF1
    r.DER_MLSE_trunc = DER_MLSE_trunc
    r.Q_budget_adj = Q_budget_adj
    r.COM_from_matlab = COM_from_matlab
    r.DER_MLSE = DER_MLSE
    r.DER_DFE = DER_DFE
    r.COM = new_com
    r.delta_com = delta_com
    r.delta_com_calc = delta_com_calc
    r.g_an = g_an
    return r
