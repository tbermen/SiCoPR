import numpy as np
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


from scipy.signal import fftconvolve
from types import SimpleNamespace


# --- inline from conv_fct (MATLAB 5371-5388) ---

# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


# --- inline from get_pdf_from_sampled_signal (MATLAB 7473-7518) ---


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    if not np.isscalar(input_vector):
        iv = np.asarray(input_vector, dtype=float).ravel()
    else:
        iv = np.array([float(input_vector)])
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


def Burst_Probability_Calc(COM_SNR_Struct, DFE_taps, param, OP):
    """Compute burst error probability (MATLAB lines 1083-1132).

    Returns (p_burst, p_error_propagation).
    """
    A_s = float(COM_SNR_Struct.A_s)
    error_propagation_noise_pdf = [COM_SNR_Struct.combined_interference_and_noise_pdf]
    error_threshold = A_s / (10 ** ((float(param.pass_threshold) - float(OP.COM_EP_margin)) / 20.0))

    pdf0 = error_propagation_noise_pdf[0]
    idx = np.where(pdf0.x >= error_threshold)[0]
    p_ep_0 = 1e-20 if len(idx) == 0 else float(np.sum(pdf0.y[idx[0]:]))
    p_error_propagation = [p_ep_0]

    sorted_abs_taps = np.sort(np.abs(np.asarray(DFE_taps).ravel()))[::-1]
    nburst = int(OP.nburst)
    ndfe = int(param.ndfe)
    for k in range(1, min(ndfe, nburst)):
        if OP.use_simple_EP_model:
            tap_val = 2.0 * A_s * float(_mmax(sorted_abs_taps))
            post_pdf = _get_pdf_from_sampled_signal(tap_val, param.levels, param.delta_y)
            new_pdf = _conv_fct(error_propagation_noise_pdf[0], post_pdf)
        else:
            tap_val = 2.0 * A_s * float(sorted_abs_taps[k - 1])
            post_pdf = _get_pdf_from_sampled_signal(tap_val, param.levels, param.delta_y)
            new_pdf = _conv_fct(error_propagation_noise_pdf[k - 1], post_pdf)
        error_propagation_noise_pdf.append(new_pdf)
        idx = np.where(new_pdf.x >= error_threshold)[0]
        p_ep_k = 1e-20 if len(idx) == 0 else float(np.sum(new_pdf.y[idx[0]:]))
        p_error_propagation.append(p_ep_k)

    p_burst = np.cumprod(p_error_propagation)
    return p_burst, p_error_propagation
