# ============================================================
# MATLAB→Python translation notes for Fract_T_FFE
# MATLAB lines: 2109–2118
# ============================================================
# iscolumn(V): if (N,1) column vector, transpose → 1-D ravel ✓
# circshift(V',[ishift,0])': shift the TRANSPOSE by ishift rows, transpose back.
#   For a 1-D V that is np.roll(V, ishift); for a 2-D V it shifts across
#   columns, which np.roll(V, ishift) does not do. See below.
# V0 = (circshift(V,ishift) + V) / 2: weighted average of current and shifted.
# skew_step=0 → V0 = (V+V)/2 = V (identity).
# skew_step must be a whole number or MATLAB's circshift errors. See below.
# ============================================================

import numpy as np


def Fract_T_FFE(V, skew_step):
    V = np.asarray(V, dtype=float)
    if V.ndim == 2 and V.shape[1] == 1:
        V = V.ravel()
    # MATLAB circshift refuses a fractional shift, so Python must too: np.roll
    # silently truncated 1.5 to 1 and answered.
    # COM Octave: Fract_T_FFE([1 2 3 4], 1.5) ->
    #   "circshift: all values of N must be integers"  (Python gave [2.5 1.5 2.5 3.5])
    if skew_step != int(skew_step):
        raise ValueError('circshift: all values of N must be integers')
    # circshift(V',[ishift,0])' shifts the transpose along its rows, i.e. a 2-D
    # V is shifted across its COLUMNS. np.roll(V, ishift) flattens instead.
    # COM Octave: Fract_T_FFE([1 2;3 4;5 6], 1) -> [1.5 1.5; 3.5 3.5; 5.5 5.5]
    #   (Python gave [[3.5,1.5],[2.5,3.5],[4.5,5.5]].)
    # For a 1-D V, V.T is V and this is the old np.roll(V, skew_step).
    V0 = (np.roll(V.T, int(skew_step), axis=0).T + V) / 2
    return V0


if __name__ == "__main__":
    V = np.array([1.0, 2.0, 3.0, 4.0])
    print("skew=0:", Fract_T_FFE(V, 0))   # [1,2,3,4]
    print("skew=1:", Fract_T_FFE(V, 1))   # [2.5,1.5,2.5,3.5]
