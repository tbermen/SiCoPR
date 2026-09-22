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
    # Both masks are taken from the ORIGINAL input: MATLAB computes
    # input<min_threshold, not clip_output<min_threshold, so with crossed
    # bounds (min>max) an element can be raised after being lowered.
    # Octave: dfe_clipper([0 1.5 3],[1 1 1],[2 2 2]) -> [2 2 1].
    # NaN compares false both ways and passes through unclipped:
    # dfe_clipper([1 NaN 3],[2 2 2],[0 0 0]) -> [1 NaN 2].
    mask_hi = inp > hi
    mask_lo = inp < lo

    # MATLAB writes max_threshold(input>max_threshold): a logical index into
    # the THRESHOLD array. It errors as soon as a true position falls past the
    # end of that array, so a scalar threshold works only while nothing beyond
    # the first element is clipped. Octave:
    #     dfe_clipper([3 1 1], 2, -9) -> [2 1 1]      (only position 1 true)
    #     dfe_clipper([1 3 1], 2, -9) -> error: max_threshold(2): out of bound 1
    # numpy would instead broadcast the scalar and return a plausible answer
    # for a call MATLAB refuses.
    # Assigning positionally rather than with a boolean mask is what makes
    # that emulation possible: numpy requires a boolean index to match the
    # array's shape exactly, while MATLAB only requires every TRUE position to
    # be in range -- so a threshold that is scalar, or simply longer than the
    # input, is legal there and a shape error here.
    # MATLAB linear indexing is column-major; identical to C order for the
    # vectors every caller passes, but order='F' keeps 2-D honest.
    out_f = out.ravel(order='F')
    for mask, thr, nm in ((mask_hi, hi, 'max_threshold'),
                          (mask_lo, lo, 'min_threshold')):
        where = np.nonzero(np.asarray(mask).ravel(order='F'))[0]
        if where.size == 0:
            continue
        if where.max() >= thr.size:
            raise IndexError(
                'dfe_clipper: %s(%d): out of bound %d -- MATLAB indexes the '
                'threshold array with the input-shaped logical mask, so it '
                'errors here rather than broadcasting.'
                % (nm, where.max() + 1, thr.size))
        out_f[where] = thr.ravel(order='F')[where]
    return out_f.reshape(inp.shape, order='F')


if __name__ == "__main__":
    import numpy as np
    x = np.array([1.0, 5.0, 3.0])
    hi = np.array([2.0, 4.0, 3.5])
    lo = np.array([0.0, 1.0, 2.0])
    print("dfe_clipper([1,5,3], [2,4,3.5], [0,1,2]) =", dfe_clipper(x, hi, lo))
    # expected: [1, 4, 3]
