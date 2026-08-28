# ============================================================
# MATLAB→Python translation notes for dfe_clipper
# MATLAB lines: 5531–5544
# ============================================================
# 1-based vs 0-based indexing: not applicable (boolean indexing only).
# Column-major vs row-major: MATLAB isrow() distinguishes 1×n from n×1.
#   Python 1-D arrays have no inherent orientation; we treat ndim==1 as
#   "row-like" (threshold stays flat) and ndim==2 with shape[1]==1 as
#   column (threshold reshaped to (-1,1)).
# MATLAB `end` keyword: not used.
# Output shape: same shape as input.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def dfe_clipper(input_arr, max_threshold, min_threshold):
    """Element-wise clip with per-element bounds.

    Clips input so that:
      input[i] > max_threshold[i]  →  output[i] = max_threshold[i]
      input[i] < min_threshold[i]  →  output[i] = min_threshold[i]
    Threshold orientation is aligned to input (row vs column).
    """
    inp = np.asarray(input_arr, dtype=float)
    hi = np.asarray(max_threshold, dtype=float)
    lo = np.asarray(min_threshold, dtype=float)

    # MATLAB isrow(input): true for 1-D or 2-D with shape[0]==1
    is_row = inp.ndim <= 1 or (inp.ndim == 2 and inp.shape[0] == 1)
    if is_row:
        hi = hi.ravel()   # (:).' in MATLAB — flatten to row-compatible 1-D
        lo = lo.ravel()
    else:
        hi = hi.ravel().reshape(-1, 1)  # (:) in MATLAB — column vector
        lo = lo.ravel().reshape(-1, 1)

    out = inp.copy()
    mask_hi = inp > hi
    mask_lo = inp < lo
    out[mask_hi] = hi[mask_hi]
    out[mask_lo] = lo[mask_lo]
    return out


if __name__ == "__main__":
    import numpy as np
    x = np.array([1.0, 5.0, 3.0])
    hi = np.array([2.0, 4.0, 3.5])
    lo = np.array([0.0, 1.0, 2.0])
    print("dfe_clipper([1,5,3], [2,4,3.5], [0,1,2]) =", dfe_clipper(x, hi, lo))
    # expected: [1, 4, 3]
