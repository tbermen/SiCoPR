import numpy as np

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


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)


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

from scipy.signal import fftconvolve
import copy
from types import SimpleNamespace


# --- inline from conv_fct (MATLAB 5371-5388) ---

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


def _conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')
    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


# --- inline from CDF_inv_ev (MATLAB 1142-1148) ---
def _CDF_inv_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    CDF = np.asarray(CDF, dtype=float)
    idx = np.where(CDF >= val)[0]
    return float(x[-1]) if len(idx) == 0 else float(x[idx[0]])


# --- inline from scalePDF (MATLAB 11268-11275) ---
def _scalePDF(pdf, scale_factor):
    pdf_out = copy.copy(pdf)
    pdf_out.Min = int(np.floor(pdf.Min * scale_factor))
    idx = np.arange(pdf_out.Min, -pdf_out.Min + 1)
    pdf_out.x = idx * pdf_out.BinSize
    pdf_out.y = np.interp(pdf_out.x, np.asarray(pdf.x) * scale_factor, np.asarray(pdf.y))
    pdf_out.y[0] = pdf_out.y[1]
    pdf_out.y[-1] = pdf_out.y[-2]
    pdf_out.y = pdf_out.y / np.sum(pdf_out.y)
    return pdf_out


# --- inline from get_pdf_from_sampled_signal (MATLAB 7473-7518) ---
def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.any(np.diff(values) < 0):
        si = np.argsort(values, kind='stable')
        values, probs = values[si], probs[si]
    values = binsize * _mround_arr(values / binsize)
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
    rvd = _mround_arr(values / pdf.BinSize).astype(int)
    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])
    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    input_vector = np.asarray(input_vector, dtype=float).ravel()
    if _mmax(np.abs(input_vector)) > BinSize:
        input_vector = input_vector[np.abs(input_vector) > BinSize]
    else:
        return _d_cpdf(BinSize, 0, 1)
    input_vector[np.abs(input_vector) < BinSize] = 0.0
    b = np.sign(input_vector)
    sort_idx = np.argsort(np.abs(input_vector), kind='stable')[::-1]
    input_vector = np.abs(input_vector[sort_idx]) * b[sort_idx]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L
    pdf = _d_cpdf(BinSize, 0, 1)
    empty_pdf = pdf
    for v in input_vector:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct(pdf, pdfn)
    return pdf


def adjust_Rx_noise_for_quantization(combined_interference_and_noise_pdf, NS, chdata,
                                      fom_result, param, OP):
    """Adjust combined PDF to include quantization noise (MATLAB lines 4858-4896).

    Returns (chdata, NS, combined_interference_and_noise_pdf).
    """
    sig_after_ctle_pdf = _get_pdf_from_sampled_signal(
        chdata[0].pulse_sampled_w_tx_ffe_ctle, param.levels, param.delta_y)
    sig_after_ctle_pdf = _conv_fct(sig_after_ctle_pdf, combined_interference_and_noise_pdf)
    sig_after_ctle_cdf = np.cumsum(sig_after_ctle_pdf.y)
    adc_clip = -float(_CDF_inv_ev(param.P_qc, sig_after_ctle_pdf, sig_after_ctle_cdf))
    ctle_signal_sigma = float(np.sqrt(np.sum((sig_after_ctle_pdf.x ** 2) * sig_after_ctle_pdf.y)))
    adc_lsb = 2.0 * adc_clip / (2 ** param.N_qb - 1)
    NS.sigma_Q = adc_lsb / np.sqrt(12.0)
    NS.sigma_before_clip = ctle_signal_sigma
    NS.peak_clip = adc_clip
    NS.p2ptosigma_clip = 2.0 * adc_clip / ctle_signal_sigma

    quantization_noise_in_pdf = copy.copy(combined_interference_and_noise_pdf)
    quantization_noise_in_pdf.y = np.zeros_like(quantization_noise_in_pdf.x, dtype=float)
    adc_ind_right = int(np.argmin(np.abs(quantization_noise_in_pdf.x - adc_lsb / 2)))
    adc_ind_left = int(np.argmin(np.abs(quantization_noise_in_pdf.x + adc_lsb / 2)))
    n_bins = adc_ind_right - adc_ind_left + 1
    quantization_noise_in_pdf.y[adc_ind_left:adc_ind_right + 1] = 1.0 / n_bins

    quantization_noise_pdf = copy.copy(combined_interference_and_noise_pdf)
    ind_center = int(np.argmin(np.abs(quantization_noise_pdf.x)))
    quantization_noise_pdf.y = np.zeros_like(quantization_noise_pdf.x, dtype=float)
    quantization_noise_pdf.y[ind_center] = 1.0

    rxffe = np.asarray(fom_result.RxFFE).ravel()
    h_rxffe = rxffe[rxffe != 0]
    cursor_tap = int(param.ffe_pre_tap_len) + 1  # r4p15p0: 1-based cursor tap position
    for irxffe_0, val in enumerate(h_rxffe):
        if (irxffe_0 + 1) != cursor_tap:
            scaled = _scalePDF(quantization_noise_in_pdf, float(np.abs(val)))
            quantization_noise_pdf = _conv_fct(quantization_noise_pdf, scaled)

    combined_interference_and_noise_pdf = _conv_fct(
        combined_interference_and_noise_pdf, quantization_noise_pdf)
    NS.quantization_noise_pdf = quantization_noise_pdf
    return chdata, NS, combined_interference_and_noise_pdf
