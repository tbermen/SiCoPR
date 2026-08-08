# ============================================================
# MATLAB→Python translation notes for comb_fct
# MATLAB lines: 5258–5285
# ============================================================
# PDF struct format: p.BinSize (bin width), p.Min (negative integer bin index),
#   p.y (probability array), p.x (voltage axis).
# p.x = (p.Min:-p.Min)*p.BinSize → symmetric voltage axis:
#   MATLAB range p.Min:-p.Min → np.arange(p.Min, -p.Min+1), e.g., -5:5 → [-5..5]
#   len(x) = -2*p.Min + 1 = 2*|p.Min| + 1
# difsz = abs(p1.Min - p2.Min) → integer shift in bins
# p1.Min == p.Min: p1 extends further left; zero-pad p2 on both sides
# p2.Min == p.Min: p2 extends further left; zero-pad p1 on both sides
# Equal Min: neither branch runs; p1.y and p2.y added directly (same length)
# MATLAB in-place array extension translated as constructing new zero array
#   and slicing in the original values: new_y[difsz:lp+difsz] = original_y
# p=p1 copies all fields → SimpleNamespace(**vars(p1)) then override Min/y/x
# ============================================================

import numpy as np
from types import SimpleNamespace


def comb_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))       # copy all fields (MATLAB p=p1)
    p.BinSize = p1.BinSize
    p.Min = min(p1.Min, p2.Min)
    difsz = int(abs(p1.Min - p2.Min))
    lp1 = len(p1.y)
    lp2 = len(p2.y)
    p1_y = np.asarray(p1.y, dtype=float).copy()
    p2_y = np.asarray(p2.y, dtype=float).copy()

    if p1.Min == p.Min:           # p1 has the smaller (more negative) Min
        new_p2 = np.zeros(lp1)
        new_p2[difsz:lp2 + difsz] = p2_y
        p2_y = new_p2
    elif p2.Min == p.Min:         # p2 has the smaller Min
        new_p1 = np.zeros(lp2)
        new_p1[difsz:lp1 + difsz] = p1_y
        p1_y = new_p1

    p.y = p1_y + p2_y
    p.x = np.arange(p.Min, -p.Min + 1) * p.BinSize   # (p.Min:-p.Min)*BinSize
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=-2, BinSize=0.1, y=np.array([1,2,3,2,1]))
    p2 = SimpleNamespace(Min=-1, BinSize=0.1, y=np.array([1,2,1]))
    out = comb_fct(p1, p2)
    print("y:", out.y)   # expect [1,3,5,3,1]
    print("x:", out.x)   # expect [-0.2,-0.1,0,0.1,0.2]
