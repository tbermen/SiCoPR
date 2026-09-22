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
        p = SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
        return p
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
        if k == 0:
            bin_idx = 0
        elif k == len(values) - 1:
            bin_idx = len(t) - 1
        else:
            bin_idx = int(np.argmin(np.abs(t - v)))
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
    if np.any(bp < 0) or np.any(bp >= len(pdf.y)):
        # pdf.x only spans rvd(1)..rvd(end), so any value that rounds outside
        # that span (i.e. `values` is not ascending) makes bin_placement fall
        # off the array and MATLAB stops.  A negative index is legal in numpy,
        # so Python wrapped round and added the probability to the wrong bin.
        # COM Octave 4p16p0: Init_PDF_Fast(E,[0 -0.2 0.3],[0.2 0.3 0.5]) with
        # BinSize=0.1 errors "pdf(-1): subscripts must be either integers
        # 1 to (2^63)-1 or logicals"; Python answered y=[0.2 0 0.3 0.5].
        raise IndexError('Init_PDF_Fast: values must be ascending')
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


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
