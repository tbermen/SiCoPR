import numpy as np
from com_functions.fn.CDF_inv_ev.py_impl import CDF_inv_ev as _CDF_inv_ev
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf
from com_functions.fn.scalePDF.py_impl import scalePDF as _scalePDF

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


from scipy.signal import fftconvolve
import copy
from types import SimpleNamespace


# --- inline from conv_fct (MATLAB 5371-5388) ---

# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


# --- inline from CDF_inv_ev (MATLAB 1142-1148) ---


# --- inline from scalePDF (MATLAB 11268-11275) ---


# --- inline from get_pdf_from_sampled_signal (MATLAB 7473-7518) ---


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    input_vector = np.asarray(input_vector, dtype=float).ravel()
    # MATLAB max([]) is [], and `if []` is false, so an empty sampled pulse
    # falls straight through to the delta pdf.  COM Octave,
    # adjust_Rx_noise_for_quantization with
    # chdata(1).pulse_sampled_w_tx_ffe_ctle = []: NS.peak_clip = 0.2, the same
    # answer a wholly sub-bin pulse gives.  np.max raised "zero-size array to
    # reduction operation maximum".  (The canonical
    # com_functions/fn/get_pdf_from_sampled_signal/py_impl.py already carries
    # this guard; this copy had not tracked it.)
    if input_vector.size and _mmax(np.abs(input_vector)) > BinSize:
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
