# ============================================================
# MATLAB→Python translation notes for conv_fct
# MATLAB lines: 5371–5388
# ============================================================
# conv2(p1.y, p2.y): for 1-D row vectors, conv2 = conv → np.convolve ✓
# p.Min = round(p1.Min+p2.Min): Min is an integer bin index; round is
#   a no-op for integer inputs but kept for fidelity.
# pMax = p.Min + len(p.y) - 1: highest bin index (output has N1+N2-1 bins)
# p.x = (p.Min*BinSize : BinSize : pMax*BinSize):
#   Python: np.arange(p.Min, pMax+1) * BinSize  (integer bin indices × BinSize)
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
    if min(a.size, b.size) >= _CONV_FFT_MIN:
        return fftconvolve(a, b)
    return np.convolve(a, b)


def conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))                     # p=p1 copies all fields
    p.Min = int(round(p1.Min + p2.Min))                 # sum of bin-index minimums
    p.y = _conv1d(p1.y, p2.y)    # conv2 on 1-D = convolve
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize       # (p.Min*BinSize:BinSize:pMax*BinSize)
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1.0, 1.0]))
    p2 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1.0, 1.0]))
    out = conv_fct(p1, p2)
    print("y:", out.y)   # [1,2,1]
    print("x:", out.x)   # [-0.2,-0.1,0]
