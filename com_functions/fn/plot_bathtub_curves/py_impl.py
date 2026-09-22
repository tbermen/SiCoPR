# ============================================================
# MATLAB→Python translation notes for plot_bathtub_curves
# MATLAB lines: 8865–8913
# ============================================================
# Inline helpers: conv_fct (5371-5388), d_cpdf (MATLAB d_cpdf).
# cursors: two-point PDF at ±max_signal with equal probability 1/2.
# MATLAB semilogy → axes.semilogy in Python.
# vbt_r: MATLAB fliplr(0.5 - cumsum(fliplr(...))) → np.flipud for 1-D arrays.
# The if 0 ... end debug block is omitted.
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
    return SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                           x=np.arange(pdf_min, -pdf_min + 1) * binsize)


def plot_bathtub_curves(hax, max_signal, sci_pdf, cci_pdf, isi_and_xtalk_pdf,
                        noise_pdf, jitt_pdf, combined_interference_and_noise_pdf, bin_size):
    cursors = _d_cpdf(bin_size, max_signal * np.array([-1.0, 1.0]), np.array([0.5, 0.5]))
    signal_and_isi_pdf = _conv_fct(cursors, sci_pdf)
    signal_and_xtalk_pdf = _conv_fct(cursors, cci_pdf)
    signal_and_channel_noise_pdf = _conv_fct(cursors, isi_and_xtalk_pdf)
    signal_and_system_noise_pdf = _conv_fct(cursors, noise_pdf)
    signal_and_system_jitt_pdf = _conv_fct(cursors, jitt_pdf)

    cursors_l = copy.copy(cursors)
    cursors_l.y = cursors_l.y.copy()
    cursors_l.y[cursors_l.x > 0] = 0

    cursors_r = copy.copy(cursors)
    cursors_r.y = cursors_r.y.copy()
    cursors_r.y[cursors_r.x < 0] = 0

    signal_and_total_noise_pdf_l = _conv_fct(cursors_l, combined_interference_and_noise_pdf)
    signal_and_total_noise_pdf_r = _conv_fct(cursors_r, combined_interference_and_noise_pdf)

    hax.semilogy(signal_and_isi_pdf.x, np.abs(np.cumsum(signal_and_isi_pdf.y) - 0.5),
                 'r', label='ISI')
    hax.semilogy(signal_and_xtalk_pdf.x, np.abs(np.cumsum(signal_and_xtalk_pdf.y) - 0.5),
                 'b', label='Xtalk')
    hax.semilogy(signal_and_channel_noise_pdf.x, np.abs(np.cumsum(signal_and_channel_noise_pdf.y) - 0.5),
                 'c', label='ISI+Xtalk')
    hax.semilogy(signal_and_system_noise_pdf.x, np.abs(np.cumsum(signal_and_system_noise_pdf.y) - 0.5),
                 'm', label='Jitter, SNR_TX,RL_M, eta_0 noise')
    hax.semilogy(signal_and_system_jitt_pdf.x, np.abs(np.cumsum(signal_and_system_jitt_pdf.y) - 0.5),
                 'g', label='Jitter noise')

    vbt_l = np.abs(0.5 - np.cumsum(signal_and_total_noise_pdf_l.y))
    vbt_r = np.flipud(0.5 - np.cumsum(np.flipud(signal_and_total_noise_pdf_r.y)))
    hax.semilogy(signal_and_total_noise_pdf_l.x, vbt_l, 'k', label='total noise PDF left')
    hax.semilogy(signal_and_total_noise_pdf_r.x, vbt_r, 'k', label='total noise PDF right')

    hax.plot(max_signal * np.array([-1.0, -1.0, 1.0, 1.0]),
             [0.5, 1e-20, 1e-20, 0.5], '--ok')
    hax.set_ylabel('Probability')
    hax.set_xlabel('volts')
    hax.legend()
