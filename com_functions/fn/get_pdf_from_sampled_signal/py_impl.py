import numpy as np
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


from scipy.signal import fftconvolve
from types import SimpleNamespace



# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


def get_pdf_from_sampled_signal(input_vector, L, BinSize, FAST_NOISE_CONV=0):
    """Create ISI PDF from sampled signal via successive delta convolutions (MATLAB lines 7473-7518).

    input_vector: ISI sample values
    L: number of PAM levels
    BinSize: PDF grid spacing
    FAST_NOISE_CONV: when set, the small (|tap|<0.001) residual taps are approximated as a
        single Gaussian (normal_dist) and convolved in once, instead of one delta-set
        convolution per tap.  MATLAB calls conv_fct_TEST here, which is UNDEFINED in the
        reference (the commented-out L7516 shows conv_fct was the intent) -> we use conv_fct.
        This is an optional speed approximation; FAST_NOISE_CONV=0 (default) is exact.
    """
    input_vector = np.asarray(input_vector, dtype=float).ravel()

    if _mmax(np.abs(input_vector)) > BinSize:
        input_vector = input_vector[np.abs(input_vector) > BinSize]
    else:
        return _d_cpdf(BinSize, 0, 1)

    input_vector[np.abs(input_vector) < BinSize] = 0.0
    b = np.sign(input_vector)
    sort_idx = np.argsort(np.abs(input_vector), kind='stable')[::-1]  # descending by |value|
    input_vector = np.abs(input_vector[sort_idx]) * b[sort_idx]

    # ---- FAST_NOISE_CONV: split off small residual taps as one Gaussian (MATLAB L7498-7502) ----
    res_pdf = None
    if FAST_NOISE_CONV:
        small = np.where(np.abs(input_vector) < 0.001)[0]
        if len(small) > 0:
            first_small = int(small[0])  # MATLAB find(...,1), 0-based here
            sig_res = float(np.linalg.norm(input_vector[first_small + 1:]))
            res_pdf = _normal_dist(sig_res, 5, BinSize)
            input_vector = input_vector[:first_small + 1]
        # (no small taps -> nothing to approximate; keep all taps exact)

    # Equation 93A-39: values uniformly spaced in [-1, 1]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L

    pdf = _d_cpdf(BinSize, 0, 1)
    empty_pdf = pdf
    for v in input_vector:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct(pdf, pdfn)

    if res_pdf is not None:  # MATLAB L7515-7517 (conv_fct_TEST -> conv_fct)
        pdf = _conv_fct(pdf, res_pdf)

    return pdf
