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
from types import SimpleNamespace


def conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))                     # p=p1 copies all fields
    p.Min = int(round(p1.Min + p2.Min))                 # sum of bin-index minimums
    p.y = np.convolve(np.asarray(p1.y, dtype=float),
                      np.asarray(p2.y, dtype=float))    # conv2 on 1-D = convolve
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize       # (p.Min*BinSize:BinSize:pMax*BinSize)
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1.0, 1.0]))
    p2 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1.0, 1.0]))
    out = conv_fct(p1, p2)
    print("y:", out.y)   # [1,2,1]
    print("x:", out.x)   # [-0.2,-0.1,0]
