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
from com_functions.fn.Bessel_Thomson_Filter.py_impl import Bessel_Thomson_Filter as _Bessel_Thomson_Filter
from com_functions.fn.Butterworth_Filter.py_impl import Butterworth_Filter as _Butterworth_Filter
from com_functions.fn.normal_dist.py_impl import normal_dist as _normal_dist
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


from scipy.signal import lfilter, fftconvolve
from types import SimpleNamespace


# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    iv = np.asarray(input_vector, dtype=float).ravel()
    if len(iv) == 0:
        return _d_cpdf(BinSize, 0, 1)
    if _mmax(np.abs(iv)) > BinSize:
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

    ipeak = int(np.where(REF_PR == _mmax(REF_PR))[0][0])  # 0-based
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

    result.FOM = float(_mmax([
        float(np.linalg.norm(ILN[im::M])) for im in range(M)
    ]))

    pdf_from_norm = _normal_dist(result.FOM, 7, BinSize)
    fit_peak = np.float64(FIT_PR[ipeak])
    # MATLAB's local `db = @(x) 20*log10(abs(x))` — the abs() is the point. The
    # ratio is NEGATIVE whenever FIT.PR at REF.PR's peak has the opposite sign
    # (an inverted or badly mismatched fit), and MATLAB then reports a finite
    # negative dB. Dropping the abs() and guarding with `FOM > 0` instead gave
    # log10 of a negative -> NaN: COM Octave returns SNR_ISI_FOM
    # -6.0602241763211184 and SNR_ISI_FOM_PDF -7.3036991416734809 on the case
    # the unit test pins, where the port returned nan for both.
    # The guards are not needed either: FOM = 0 gives MATLAB's 20*log10(Inf) =
    # Inf, and numpy's float64 division gives the same Inf.
    with np.errstate(divide='ignore', invalid='ignore'):
        result.SNR_ISI_FOM = float(
            20 * np.log10(np.abs(fit_peak / np.float64(result.FOM))))
        result.SNR_ISI_FOM_PDF = float(
            20 * np.log10(np.abs(fit_peak / np.float64(result.FOM_PDF))))

    print(f'SNR ISI FOM rms = {result.SNR_ISI_FOM:.6g} dB;   '
          f'SNR ISI FOM PDF = {result.SNR_ISI_FOM_PDF:.6g} dB')

    return result
