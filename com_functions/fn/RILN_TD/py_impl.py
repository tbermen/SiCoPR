# ============================================================
# MATLAB→Python translation notes for RILN_TD
# MATLAB lines: 4226–4305
# ============================================================
# s21_to_impulse_DC: calls the real top-level function (L230/L241), which
# handles non-zero IL. A private inlined copy that raised
# NotImplementedError for non-zero IL used to sit here; it had no callers
# once the real one was wired, and its raise was the last thing making the
# assembler report this function as unimplemented. Removed 2026-08-30.
# All filter helpers (Bessel_Thomson, Butterworth, Tukey) inlined.
# H_tw: MATLAB overrides Tukey with ones (H_tw=ones(1,length(f))) after computing it.
# filter(ones(1,M), 1, FIR) → scipy.signal.lfilter(np.ones(M), [1.0], FIR).
# ipeak: 0-based argmax of REF.PR.
# range_end: 0-based; MATLAB 1-based range → Python slice ipeak:range_end+1.
# im loop: MATLAB 1:M → Python 0:M-1; ILN[im::M] for each phase.
# rms: sqrt(pdf.y @ pdf.x^2) * sqrt(2) (MATLAB row*col product).
# cdf.y: cumsum(pdf.y).
# FOM_PDF: -cdf.x at first index where cdf.y >= specBER.
# normal_dist inlined.
# Plotting (MATLAB figure calls) replaced by print for debug.
# ============================================================

import math
import numpy as np
from scipy.signal import lfilter, fftconvolve
from types import SimpleNamespace


_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]



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


def _bessel_poly(n):
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        a[ii] = (math.factorial(2 * n - ii)
                 / (2 ** (n - ii) * math.factorial(ii) * math.factorial(n - ii)))
    return a


def _Butterworth_Filter(param, f, use_BW):
    if not use_BW:
        return np.ones(len(f))
    s = 1j * np.asarray(f, dtype=float) / (param.fb_BW_cutoff * param.fb)
    return 1.0 / np.polyval(_BW_POLY, s)


def _Bessel_Thomson_Filter(param, f, use_BT):
    if not use_BT:
        return np.ones(len(f))
    a = _bessel_poly(param.BTorder)
    acoef = a[::-1]
    s = 1j * np.asarray(f, dtype=float) / (param.fb_BT_cutoff * param.fb)
    return a[0] / np.polyval(acoef, s)


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.any(np.diff(values) < 0):
        si = np.argsort(values, kind='stable')
        values, probs = values[si], probs[si]
    values = binsize * np.round(values / binsize)
    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize
    pdf_y = np.zeros(len(t))
    for k, (v, prob) in enumerate(zip(values, probs)):
        if k == 0:
            bin_idx = 0
        elif k == len(values) - 1:
            bin_idx = len(t) - 1
        else:
            bin_idx = int(np.argmin(np.abs(t - v)))
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
    sort_idx = np.argsort(np.abs(iv), kind='stable')[::-1]
    iv = np.abs(iv[sort_idx]) * b[sort_idx]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L
    pdf = _d_cpdf(BinSize, 0, 1)
    empty_pdf = pdf
    for v in iv:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct(pdf, pdfn)
    return pdf


def _normal_dist(sigma, nsigma, binsize):
    pdf = SimpleNamespace()
    pdf.BinSize = binsize
    pdf.Min = -round(2 * nsigma * sigma / binsize)
    pdf.x = np.arange(pdf.Min, -pdf.Min + 1) * binsize
    eps = np.finfo(float).eps
    pdf.y = np.exp(-pdf.x ** 2 / (2 * sigma ** 2 + eps))
    pdf.y = pdf.y / np.sum(pdf.y)
    return pdf


def RILN_TD(sdd21, RIL, faxis_f2, OP, param, A_T=None):
    sdd21 = np.asarray(sdd21, dtype=complex).ravel()
    RIL = np.asarray(RIL, dtype=complex).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    print('computing TD_RILN...')
    result = SimpleNamespace()

    if not hasattr(OP, 'transmitter_transition_time'):
        raise AttributeError('OP.transmitter_transition_time required for RILN_TD')

    H_bt = _Bessel_Thomson_Filter(param, faxis_f2, True)
    H_bw = _Butterworth_Filter(param, faxis_f2, True)
    H_t = np.exp(-(np.pi * faxis_f2 / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    # MATLAB overrides H_tw with ones after computing it
    H_tw = np.ones(len(faxis_f2))

    # MATLAB L4249: real top-level s21_to_impulse_DC (interp_Sparam implemented).
    REF_FIR, REF_t, REF_caus_dB, REF_trunc_dB = s21_to_impulse_DC(
        sdd21 * H_bw * H_t * H_tw, faxis_f2, param.sample_dt, OP, param)
    result.REF = SimpleNamespace(
        FIR=REF_FIR, t=REF_t,
        causality_correction_dB=REF_caus_dB, truncation_dB=REF_trunc_dB)

    M = int(param.samples_per_ui)
    result.REF.PR = lfilter(np.ones(M), [1.0], REF_FIR)

    # MATLAB L4254: real top-level s21_to_impulse_DC.
    FIT_FIR, FIT_t, FIT_caus_dB, FIT_trunc_dB = s21_to_impulse_DC(
        RIL * H_bw * H_t * H_tw, faxis_f2, param.sample_dt, OP, param)
    result.FIT = SimpleNamespace(
        FIR=FIT_FIR, t=FIT_t,
        causality_correction_dB=FIT_caus_dB, truncation_dB=FIT_trunc_dB)
    result.FIT.PR = lfilter(np.ones(M), [1.0], FIT_FIR)

    REF_PR = result.REF.PR
    FIT_PR = result.FIT.PR

    ipeak = int(np.where(REF_PR == np.max(REF_PR))[0][0])  # 0-based
    NrangeUI = 1000
    range_end = min(ipeak + M * NrangeUI,
                    min(len(FIT_FIR), len(REF_FIR)) - 1)
    irange = slice(ipeak, range_end + 1)

    result.ILN = FIT_PR[irange] - REF_PR[irange]
    result.t = FIT_t[irange]
    result.FOM = -np.inf
    result.FOM_PDF = -np.inf

    ILN = result.ILN
    BinSize = float(OP.BinSize)
    specBER = float(param.specBER)
    rms_fom = -np.inf
    best_pdf = None

    for im in range(M):  # im=0..M-1; MATLAB im=1..M → same samples via im::M vs im-1::M
        samples = ILN[im::M]
        pdf = _get_pdf_from_sampled_signal(samples, int(param.levels), BinSize)
        pdf_y = np.asarray(pdf.y, dtype=float)
        pdf_x = np.asarray(pdf.x, dtype=float)
        rms = float(np.sqrt(np.dot(pdf_y, pdf_x ** 2)) * np.sqrt(2))
        cdf_y = np.cumsum(pdf_y)
        if rms > rms_fom:
            rms_fom = rms
            idx_ber = np.where(cdf_y >= specBER)[0]
            if len(idx_ber) > 0:
                result.FOM_PDF = float(-pdf_x[idx_ber[0]])
            result.PDF = pdf

    result.FOM = float(np.max([
        float(np.linalg.norm(ILN[im::M])) for im in range(M)
    ]))

    pdf_from_norm = _normal_dist(result.FOM, 7, BinSize)
    fit_peak = float(FIT_PR[ipeak])
    result.SNR_ISI_FOM = float(20 * np.log10(fit_peak / result.FOM)) if result.FOM > 0 else np.inf
    result.SNR_ISI_FOM_PDF = (float(20 * np.log10(fit_peak / result.FOM_PDF))
                               if result.FOM_PDF > 0 else np.inf)

    print(f'SNR ISI FOM rms = {result.SNR_ISI_FOM:.6g} dB;   '
          f'SNR ISI FOM PDF = {result.SNR_ISI_FOM_PDF:.6g} dB')

    return result
