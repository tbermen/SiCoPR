# ============================================================
# MATLAB→Python translation notes for conv_fct_MeanNotZero
# MATLAB lines: 5389–5404
# ============================================================
# Identical logic to conv_fct — different function name only.
# Name indicates use for PDFs whose mean may be non-zero (the Min bin
# index can be non-symmetric), whereas conv_fct targets symmetric PDFs.
# Both use the same convolution formula: p.Min = round(p1.Min+p2.Min)
# and x axis (p.Min*BinSize : BinSize : pMax*BinSize).
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
    if min(a.size, b.size) >= _CONV_FFT_MIN:
        return fftconvolve(a, b)
    return np.convolve(a, b)


def conv_fct_MeanNotZero(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=2, BinSize=0.1, y=np.array([0.6, 0.4]))
    p2 = SimpleNamespace(Min=1, BinSize=0.1, y=np.array([0.7, 0.3]))
    out = conv_fct_MeanNotZero(p1, p2)
    print("Min:", out.Min, "x:", out.x, "y:", out.y)
