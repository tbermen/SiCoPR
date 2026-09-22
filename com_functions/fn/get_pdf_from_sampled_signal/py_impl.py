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


def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's."""
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


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


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        p = SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
        return p
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
    p = SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                        x=np.arange(pdf_min, -pdf_min + 1) * binsize)
    return p


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


def _conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')
    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


def _normal_dist(sigma, nsigma, binsize):
    """MATLAB normal_dist (L8410-8415): discretised zero-mean Gaussian PDF."""
    eps = np.finfo(float).eps
    p = SimpleNamespace()
    p.BinSize = binsize
    p.Min = -_mround(2 * nsigma * sigma / binsize)
    p.x = np.arange(p.Min, -p.Min + 1) * binsize
    p.y = np.exp(-p.x ** 2 / (2 * sigma ** 2 + eps))
    p.y = p.y / np.sum(p.y)
    return p


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
