# ============================================================
# MATLAB→Python translation notes for FFE
# MATLAB lines: 2026–2048
# ============================================================
# iscolumn(V): True when V is (N,1) → transpose to row: V=V.'
#   Python: if V.ndim==2 and V.shape[1]==1, ravel to 1-D ✓
# ishift = (i-1-cmx)*spui (MATLAB, 1-based i)
#         = (i-cmx)*spui   (Python, 0-based i)
#   because i_MATLAB = i_Python + 1, so (i_M-1-cmx) = (i_P-cmx) ✓
# circshift(V',[ishift,0]): shift V TRANSPOSED by ishift rows.
#   Positive ishift → move elements DOWN (index increases), wraps at end.
#   np.roll(V.T, ishift, axis=0): for a 1-D V this is exactly np.roll(V,ishift);
#   for a 2-D V it is the transpose MATLAB actually shifts. See below.
# ishift must be a whole number or MATLAB's circshift errors. See below.
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
            # MATLAB circshift refuses a fractional shift, so Python must too:
            # a non-integer spui made np.roll truncate 1.5 to 1 and answer.
            # COM Octave: FFE([1 1],0,1.5,[1 2 3 4]) ->
            #   "circshift: all values of N must be integers"
            # (Python returned [5 3 5 7].)
            if ishift != int(ishift):
                raise ValueError('circshift: all values of N must be integers')
            # circshift(V',[ishift,0]) shifts the TRANSPOSE along its rows, so
            # a 2-D V is shifted across its columns and comes back transposed.
            # np.roll(V, ishift) flattens a 2-D V and rolls the flat buffer.
            # COM Octave: FFE([0.5 1],1,1,[1 2;3 4;5 6]) -> 2x3
            #   [2 5 8; 2.5 5.5 8.5]   (Python gave 3x2 [[2,3.5],[5,6.5],[8,6.5]])
            # For a 1-D V, V.T is V and this is the old np.roll(V, ishift).
            V0 = np.roll(V.T, int(ishift), axis=0) * c + V0
    return V0


if __name__ == "__main__":
    C = np.array([0.5, 1.0])
    V = np.array([1.0, 2.0, 3.0, 4.0])
    print("FFE:", FFE(C, cmx=1, spui=1, V=V))
