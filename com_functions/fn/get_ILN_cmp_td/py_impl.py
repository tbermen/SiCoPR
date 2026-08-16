# ============================================================
# MATLAB→Python translation notes for get_ILN_cmp_td
# MATLAB lines: 6270–6378
# ============================================================
# Computes complex IL fitting and TD ILN.
# Polynomial fit: fmbg = [ones sdd21, sqrt(f)*sdd21, f*sdd21, f^2*sdd21] (column matrix).
# LGw = (sdd21 * log(abs(sdd21)) + 1j*unwrap(angle(sdd21))) transposed.
# alpha = pinv(fmbg) @ LGw → least-squares.
# efit_C = sum(alpha_i * basis_i); FIT = exp(efit_C).
# efit = dB(FIT); ILN = dB(sdd21) - efit.
# TD: calls the real s21_to_impulse_DC for non-zero sdd21 (eps-valued path for all-zero).
# ipeak: 0-based argmax of TD_ILN.REF.PR.
# Loop im=0..M-1: norm, pdf, cdf → FOM_PDF.
# ============================================================

import numpy as np
from scipy.signal import lfilter, fftconvolve
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
    if min(a.size, b.size) >= _CONV_FFT_MIN:
        return fftconvolve(a, b)
    return np.convolve(a, b)


def _s21_to_impulse_DC_zero(freq_array, time_step, OP, param):
    """Zero-input path of s21_to_impulse_DC (eps-valued output)."""
    freq_array = np.asarray(freq_array, dtype=float)
    fmax = 1.0 / time_step / 2.0
    freq_step = float(freq_array[2] - freq_array[1]) if len(freq_array) > 2 else float(freq_array[1] - freq_array[0])
    n_steps = round(fmax / freq_step)
    fout = np.arange(0, n_steps + 1) * (fmax / n_steps)
    IL_interp = np.full(len(fout), np.finfo(float).eps, dtype=complex)
    IL_symmetric = np.concatenate([
        [np.real(IL_interp[0])],
        IL_interp[1:-1],
        [np.real(IL_interp[-1])],
        np.conj(IL_interp[1:-1])[::-1]
    ])
    impulse_response = np.real(np.fft.ifft(IL_symmetric))
    L = len(impulse_response)
    t_base = np.arange(L) / (freq_step * L)
    ir_peak = np.max(np.abs(impulse_response))
    thresh = getattr(OP, 'impulse_response_truncation_threshold', 1e-7)
    last_arr = np.where(np.abs(impulse_response) > ir_peak * thresh)[0]
    ir_last = int(last_arr[-1]) if len(last_arr) > 0 else L - 1
    voltage = impulse_response[:ir_last + 1]
    t_out = t_base[:ir_last + 1]
    return voltage, t_out, 0.0, -np.inf


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.any(np.diff(values) < 0):
        si = np.argsort(values)
        values, probs = values[si], probs[si]
    values = binsize * np.round(values / binsize)
    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize
    pdf_y = np.zeros(len(t))
    for k, (v, prob) in enumerate(zip(values, probs)):
        bin_idx = (0 if k == 0 else len(t) - 1 if k == len(values) - 1
                   else int(np.argmin(np.abs(t - v))))
        pdf_y[bin_idx] += prob
    pdf_y = pdf_y / np.sum(pdf_y)
    support = np.where(pdf_y > 0)[0]
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = t_start + int(support[0])
    return SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                           x=np.arange(pdf_min, -pdf_min + 1) * binsize)


def _Init_PDF_Fast(EmptyPDF, values, probs):
    pdf = SimpleNamespace(**vars(EmptyPDF))
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    rvd = np.round(values / pdf.BinSize).astype(int)
    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])
    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


def _conv_fct(p1, p2):
    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    iv = np.asarray(input_vector, dtype=float).ravel()
    if len(iv) == 0:
        return _d_cpdf(BinSize, 0, 1)
    if np.max(np.abs(iv)) > BinSize:
        iv = iv[np.abs(iv) > BinSize]
    else:
        return _d_cpdf(BinSize, 0, 1)
    iv[np.abs(iv) < BinSize] = 0.0
    b = np.sign(iv)
    sort_idx = np.argsort(np.abs(iv))[::-1]
    iv = np.abs(iv[sort_idx]) * b[sort_idx]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L
    pdf = _d_cpdf(BinSize, 0, 1)
    empty_pdf = pdf
    for v in iv:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct(pdf, pdfn)
    return pdf


def get_ILN_cmp_td(sdd21, faxis_f2, OP, param, A_T=None):
    """Complex IL fitting and TD ILN (MATLAB lines 6270-6378).

    sdd21: complex S21 (row or column vector).
    Returns (ILN, efit, TD_ILN).
    ILN and efit are in dB. TD_ILN contains time-domain struct.
    For non-zero sdd21: uses the real s21_to_impulse_DC for the TD section.
    """
    if A_T is None:
        A_T = 1.0

    sdd21 = np.asarray(sdd21, dtype=complex).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    # Override OP settings for this function (per MATLAB L6279-6280, 6307)
    OP_copy = SimpleNamespace(**vars(OP))
    OP_copy.impulse_response_truncation_threshold = 1e-7
    OP_copy.interp_sparam_mag = 'trend_to_DC'
    OP_copy.interp_sparam_phase = 'interp_to_DC'

    # Polynomial basis for complex log fitting
    f = faxis_f2
    fmbg = np.column_stack([
        sdd21,
        np.sqrt(f) * sdd21,
        f * sdd21,
        f ** 2 * sdd21,
    ])  # shape (N, 4)

    unwraplog = np.log(np.abs(sdd21) + np.finfo(float).eps) + 1j * np.unwrap(np.angle(sdd21))
    LGw = (sdd21 * unwraplog).reshape(-1, 1)

    # Least-squares: alpha = pinv(fmbg) @ LGw
    alpha, _, _, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)
    alpha = alpha.ravel()

    efit_C = (alpha[0] + alpha[1] * np.sqrt(f) + alpha[2] * f + alpha[3] * f ** 2)
    FIT = np.exp(efit_C)

    dB = lambda x: 20.0 * np.log10(np.abs(x) + np.finfo(float).eps)
    efit = dB(np.abs(FIT))
    ILN = dB(sdd21) - efit

    print('computing TD_ILN (dB) ...', end='')

    M = int(param.samples_per_ui)
    BinSize = float(getattr(OP, 'BinSize', 1e-4))
    specBER = float(param.specBER)

    # TX/BT/BW filters (MATLAB L6309-6313). H_bt forced use_BT=1; H_bw computed
    # but NOT used in the s21_to_impulse_DC call (matches MATLAB). H_tw = ones.
    H_bt = Bessel_Thomson_Filter(param, f, 1)
    H_bw = Butterworth_Filter(param, f, 1)
    H_t = np.exp(-(np.pi * f / 1e9 * float(OP.transmitter_transition_time) / 1.6832) ** 2)  # Eq 93A-46
    H_tw = np.ones(len(f))

    TD_ILN = SimpleNamespace()

    # REF (from sdd21) — MATLAB L6315-6318: s21_to_impulse_DC(sdd21.*H_bt.*H_t.*H_tw)
    if np.all(sdd21 == 0):
        fir_r, t_r, caus_r, trunc_r = _s21_to_impulse_DC_zero(f, float(param.sample_dt), OP_copy, param)
    else:
        fir_r, t_r, caus_r, trunc_r = s21_to_impulse_DC(
            sdd21 * H_bt * H_t * H_tw, f, float(param.sample_dt), OP_copy, param)

    TD_ILN.REF = SimpleNamespace(FIR=fir_r, t=t_r,
                                  causality_correction_dB=caus_r, truncation_dB=trunc_r)
    TD_ILN.REF.PR = lfilter(np.ones(M), [1.0], fir_r)

    # FIT (from complex fitted response) — MATLAB L6321-6324: s21_to_impulse_DC(FIT.*H_bt.*H_t.*H_tw)
    if np.all(sdd21 == 0):
        fir_f, t_f, caus_f, trunc_f = _s21_to_impulse_DC_zero(f, float(param.sample_dt), OP_copy, param)
    else:
        fir_f, t_f, caus_f, trunc_f = s21_to_impulse_DC(
            FIT * H_bt * H_t * H_tw, f, float(param.sample_dt), OP_copy, param)

    TD_ILN.FIT = SimpleNamespace(FIR=fir_f, t=t_f,
                                  causality_correction_dB=caus_f, truncation_dB=trunc_f)
    TD_ILN.FIT.PR = lfilter(np.ones(M), [1.0], fir_f)

    ipeak = int(np.where(TD_ILN.REF.PR == np.max(TD_ILN.REF.PR))[0][0])
    range_end = min(len(TD_ILN.REF.PR), len(TD_ILN.FIT.PR))
    irange = slice(ipeak, range_end)

    TD_ILN.ILN = TD_ILN.FIT.PR[irange] - TD_ILN.REF.PR[irange]
    TD_ILN.t = TD_ILN.FIT.t[irange] if len(TD_ILN.FIT.t) > ipeak else TD_ILN.FIT.t
    TD_ILN.FOM = -np.inf
    TD_ILN.FOM_PDF = -np.inf

    rms_fom = -np.inf
    for im in range(M):
        samples = TD_ILN.ILN[im::M]
        TD_ILN.FOM = max(float(TD_ILN.FOM), float(np.linalg.norm(samples)))
        pdf = _get_pdf_from_sampled_signal(samples, int(param.levels), BinSize)
        pdf_y = np.asarray(pdf.y, dtype=float)
        pdf_x = np.asarray(pdf.x, dtype=float)
        rms = float(np.sqrt(np.dot(pdf_y, pdf_x ** 2)) * np.sqrt(2))
        cdf_y = np.cumsum(pdf_y)
        if rms > rms_fom:
            rms_fom = rms
            idx_ber = np.where(cdf_y >= specBER)[0]
            if len(idx_ber) > 0:
                TD_ILN.FOM_PDF = float(-pdf_x[idx_ber[0]])
            TD_ILN.PDF = pdf

    # MATLAB db = @(x) 20*log10(abs(x)); the abs() handles a negative fit_peak
    # (FIT.PR at the REF peak index can be < 0 -> bare log10 gives NaN).
    fit_peak = float(TD_ILN.FIT.PR[ipeak])
    TD_ILN.SNR_ISI_FOM = (float(20.0 * np.log10(abs(fit_peak / TD_ILN.FOM)))
                           if TD_ILN.FOM != 0 else np.inf)
    TD_ILN.SNR_ISI_FOM_PDF = (float(20.0 * np.log10(abs(fit_peak / TD_ILN.FOM_PDF)))
                               if TD_ILN.FOM_PDF != 0 else np.inf)
    print(f'{TD_ILN.SNR_ISI_FOM_PDF:.6g} dB')

    return ILN, efit, TD_ILN
