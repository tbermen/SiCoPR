# ============================================================
# MATLAB→Python translation notes for Fract_T_FFE
# MATLAB lines: 2109–2118
# ============================================================
# iscolumn(V): if (N,1) column vector, transpose → 1-D ravel ✓
# circshift(V',[ishift,0])': shift column by ishift, then transpose back.
#   For 1-D Python array: np.roll(V, ishift) ✓ (see FFE translation notes)
# V0 = (circshift(V,ishift) + V) / 2: weighted average of current and shifted.
# skew_step=0 → V0 = (V+V)/2 = V (identity).
# ============================================================

import numpy as np


def Fract_T_FFE(V, skew_step):
    V = np.asarray(V, dtype=float)
    if V.ndim == 2 and V.shape[1] == 1:
        V = V.ravel()
    V0 = (np.roll(V, skew_step) + V) / 2
    return V0


if __name__ == "__main__":
    V = np.array([1.0, 2.0, 3.0, 4.0])
    print("skew=0:", Fract_T_FFE(V, 0))   # [1,2,3,4]
    print("skew=1:", Fract_T_FFE(V, 1))   # [2.5,1.5,2.5,3.5]
