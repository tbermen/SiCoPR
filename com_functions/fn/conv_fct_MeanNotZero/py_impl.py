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
from types import SimpleNamespace


def conv_fct_MeanNotZero(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = np.convolve(np.asarray(p1.y, dtype=float),
                      np.asarray(p2.y, dtype=float))
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=2, BinSize=0.1, y=np.array([0.6, 0.4]))
    p2 = SimpleNamespace(Min=1, BinSize=0.1, y=np.array([0.7, 0.3]))
    out = conv_fct_MeanNotZero(p1, p2)
    print("Min:", out.Min, "x:", out.x, "y:", out.y)
