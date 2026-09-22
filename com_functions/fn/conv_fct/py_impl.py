# ============================================================
# MATLAB→Python translation notes for conv_fct
# MATLAB lines: 5371–5388
# ============================================================
# conv2(p1.y, p2.y): for 1-D row vectors, conv2 = conv → np.convolve ✓
# p.Min = round(p1.Min+p2.Min): MATLAB round is half AWAY FROM ZERO, Python's
#   round() is half-to-even. See _mround below.
# pMax = p.Min + len(p.y) - 1: highest bin index (output has N1+N2-1 bins)
# p.x = (p.Min*BinSize : BinSize : pMax*BinSize): a floating-point colon, which
#   is NOT (p.Min:pMax)*BinSize. See _colon_x below.
# p=p1 copies all fields; we then override Min, y, x.
# ============================================================

import numpy as np
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


def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is half-to-even.

    COM Octave: p1.Min=0.5, p2.Min=0 -> p.Min=1  (Python round() gives 0)
                p1.Min=-0.5         -> p.Min=-1  (Python round() gives 0)
                p1.Min=2.5          -> p.Min=3   (Python round() gives 2)
    """
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    return int(round(x))


def _colon_x(pmin, pmax, binsize):
    """MATLAB `pmin*binsize : binsize : pmax*binsize`.

    The MATLAB comment in conv_fct_MeanNotZero calls this "equivalent to
    (p.Min:p.Min+length(p.y)-1)*p.BinSize", and it is not: the colon
    accumulates from the first element as a+k*d and pins the last element to
    the stated limit, while (a:b)*d forms each element as one product. They
    differ by 1 ulp on most bins.

    COM Octave: p.Min=-7, BinSize=1e-5, 4 bins ->
      [-7.0000000000000007e-05, -6.0000000000000008e-05,
       -5.0000000000000009e-05, -4.0000000000000003e-05]
    where (np.arange(-7,-3)*1e-5)[2] is -5e-05 instead. Swept over 7920
    (Min, length, BinSize) combinations against Octave: the product form got
    249387 of 1013684 elements wrong (24.6%, all by 1 ulp); this form got none.

    The last element is pinned to the limit only when accumulation overshoots
    it, which is what the colon does -- pinning unconditionally is wrong, e.g.
    COM Octave: p.Min=-3, BinSize=0.1, 2 bins ->
      [-0.30000000000000004, -0.20000000000000004]   (not ..., -0.2)

    Known gap: in 38 of those 7920 combinations Octave's colon yields one
    element FEWER than length(p.y), so p.x is shorter than p.y. That is not
    reproduced here (nor by the old form); it needs Octave's fuzzy element
    count, and every rule tried for it broke far more cases than it fixed.
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


def conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))                     # p=p1 copies all fields
    p.Min = _mround(p1.Min + p2.Min)                    # sum of bin-index minimums
    p.y = _conv1d(p1.y, p2.y)    # conv2 on 1-D = convolve
    pMax = p.Min + len(p.y) - 1
    p.x = _colon_x(p.Min, pMax, p.BinSize)   # (p.Min*BinSize:BinSize:pMax*BinSize)
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1.0, 1.0]))
    p2 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1.0, 1.0]))
    out = conv_fct(p1, p2)
    print("y:", out.y)   # [1,2,1]
    print("x:", out.x)   # [-0.2,-0.1,0]
