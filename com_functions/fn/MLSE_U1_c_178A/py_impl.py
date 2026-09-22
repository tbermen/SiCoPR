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

def _mround_arr(x):
    """MATLAB round() on an array: halves go away from zero, where np.round
    takes them to even.

    Only exact ties are corrected. Adding 0.5 and truncating would be wrong:
    0.49999999999999994 + 0.5 is exactly 1.0 in double precision, so that form
    rounds the largest double below a half up to 1 where MATLAB gives 0.
    """
    x = np.asarray(x, dtype=float)
    tie = np.abs(x - np.trunc(x)) == 0.5
    return np.where(tie, np.trunc(x) + np.copysign(1.0, x), np.round(x))


def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's.

    COM Octave: p1.Min=0.5, p2.Min=0 -> p.Min=1 (Python round() gives 0).
    """
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    return int(round(x))

from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
from scipy.special import erfc
from types import SimpleNamespace



# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.
_CONV_FFT_MIN = 128


def _conv1d(a, b):
    """Convolve two 1-D PDFs, choosing direct or FFT by operand size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # conv2 with an empty operand returns empty; np.convolve raises instead.
    # COM Octave: p1.y=[1 2 3], p2.y=[] -> p.y is 0x0, p.x is 1x0, p.Min=-1.
    if a.size == 0 or b.size == 0:
        return np.zeros(0)
    if min(a.size, b.size) >= _CONV_FFT_MIN:
        return fftconvolve(a, b)
    return np.convolve(a, b)


def _colon_x(pmin, pmax, binsize):
    """MATLAB `pmin*binsize : binsize : pmax*binsize`.

    The colon accumulates from the first element as a+k*d and pins the last
    element to the stated limit, where (pmin:pmax)*binsize forms each element
    as one product. Swept over 7920 (Min, length, BinSize) combinations against
    Octave, the product form got 24.6% of the elements wrong, all by 1 ulp;
    this form got none. The last element is pinned only when accumulation
    overshoots the limit -- pinning unconditionally is wrong, e.g. COM Octave
    Min=-3, BinSize=0.1, 2 bins -> [-0.30000000000000004, -0.20000000000000004].
    """
    n = pmax - pmin + 1
    if n <= 0:
        return np.zeros(0)
    a = pmin * binsize
    b = pmax * binsize
    x = a + np.arange(n) * binsize
    if (binsize > 0 and x[-1] > b) or (binsize < 0 and x[-1] < b):
        x[-1] = b
    return x


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.size(probs) < np.size(values):
        # MATLAB reads probs(k) for k = 1..length(values); a short probs is an
        # out-of-bound error, not a shorter answer.  zip() below would stop at
        # the shorter of the two and silently normalise whatever it collected.
        # COM Octave 4p16p0: d_cpdf(1,[-1 0 1],[0.5 0.5]) errors
        # "probs(3): out of bound 2 (dimensions are 1x2)".
        raise IndexError('d_cpdf: probs is shorter than values')
    # ~issorted(values): MATLAB requires every element <= the next, which is
    # false as soon as a NaN is present.  np.diff(values) < 0 is False across a
    # NaN, so that form calls [-1 NaN 1] sorted where MATLAB does not.
    if not np.all(values[:-1] <= values[1:]):
        si = np.argsort(values, kind='stable')
        values, probs = values[si], probs[si]
    values = binsize * _mround_arr(values / binsize)
    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize
    pdf_y = np.zeros(len(t))
    for k, (v, prob) in enumerate(zip(values, probs)):
        bin_idx = 0 if k == 0 else (len(t) - 1 if k == len(values) - 1
                                     else int(np.argmin(np.abs(t - v))))
        pdf_y[bin_idx] += prob
    pdf_y = pdf_y / np.sum(pdf_y)

    if np.any(pdf_y < 0):
        raise ValueError('PDF must be real and nonnegative')
    # find(pdf.y) selects *nonzero*, and NaN counts as nonzero.  `> 0` drops
    # NaN, so an all-zero or NaN-bearing probs vector left the support empty
    # and raised instead of answering.  COM Octave 4p16p0:
    # d_cpdf(1,[-1 0 1],[0.5 NaN 0.5]) returns Min=-1, y=[NaN NaN NaN].
    support = np.where(pdf_y != 0)[0]
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = t_start + int(support[0])
    return SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                           x=np.arange(pdf_min, -pdf_min + 1) * binsize)


def _conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')
    p = SimpleNamespace(**vars(p1))
    p.Min = _mround(p1.Min + p2.Min)   # MATLAB round: halves go away from zero
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    # (p.Min*BinSize : BinSize : pMax*BinSize) -- a floating-point colon, which
    # is NOT (p.Min:pMax)*BinSize.
    p.x = _colon_x(p.Min, pMax, p.BinSize)
    return p


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


def _scaleCDF(pdf, delta_com, DER0, A_s):
    P = np.cumsum(np.asarray(pdf.y, dtype=float))
    ider0 = int(np.argmax(P >= DER0))
    scale_factor = 1.0 / 10.0 ** (-delta_com / 20.0)
    pdf_out = _scale_pdf(pdf, scale_factor)
    cdf_out = np.cumsum(pdf_out.y)
    return pdf_out, cdf_out, scale_factor


def _CDF_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    cdf = np.asarray(CDF, dtype=float)
    hit = x >= -val
    if not hit.any():
        # find() is empty, so MATLAB's CDF(index) is an empty 1x0 -- there is
        # no value to return.  np.argmax on an all-False mask answers 0, which
        # would hand back CDF(1) as though it were the crossing.
        raise IndexError('CDF_ev: no PDF.x >= -val')
    index = int(np.argmax(hit))
    return float(cdf[index])


def _CDF_inv_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    cdf = np.asarray(CDF, dtype=float)
    indices = np.where(cdf >= val)[0]
    return float(x[indices[0]]) if len(indices) > 0 else float(x[-1])


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
