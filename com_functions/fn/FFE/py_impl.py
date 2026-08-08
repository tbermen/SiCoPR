# ============================================================
# MATLAB→Python translation notes for FFE
# MATLAB lines: 2026–2048
# ============================================================
# iscolumn(V): True when V is (N,1) → transpose to row: V=V.'
#   Python: if V.ndim==2 and V.shape[1]==1, ravel to 1-D ✓
# ishift = (i-1-cmx)*spui (MATLAB, 1-based i)
#         = (i-cmx)*spui   (Python, 0-based i)
#   because i_MATLAB = i_Python + 1, so (i_M-1-cmx) = (i_P-cmx) ✓
# circshift(V',[ishift,0]): shift column by ishift rows.
#   Positive ishift → move elements DOWN (index increases), wraps at end.
#   np.roll(V, ishift): positive shift → elements move to HIGHER indices ✓
# V0 accumulates as array after first non-zero tap (0+array→array).
# ============================================================

import numpy as np


def FFE(C, cmx, spui, V):
    C = np.asarray(C, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 2 and V.shape[1] == 1:
        V = V.ravel()                   # iscolumn(V) → V=V.' equivalent

    V0 = 0.0
    for i, c in enumerate(C):          # 0-based i
        if c != 0:
            ishift = (i - cmx) * spui   # MATLAB: (i-1-cmx)*spui with 1-based i
            V0 = np.roll(V, ishift) * c + V0
    return V0


if __name__ == "__main__":
    C = np.array([0.5, 1.0])
    V = np.array([1.0, 2.0, 3.0, 4.0])
    print("FFE:", FFE(C, cmx=1, spui=1, V=V))
