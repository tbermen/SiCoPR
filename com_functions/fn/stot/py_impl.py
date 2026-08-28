# ============================================================
# MATLAB→Python translation notes for stot
# MATLAB lines: 11276–11282
# ============================================================
# 1-based vs 0-based: s_params(1,1,:) → s[0,0,:]; (1,2)→[0,1]; etc.
# Input shape: (2,2) for a single frequency, or (2,2,N) for N frequencies.
#   Both are handled by squeezing to 1-D frequency vectors internally.
# Output shape: same as input (2,2) or (2,2,N).
# MATLAB [A B; C D] where A,B,C,D are 1×1×N → (2,2,N) result.
# eps guard: s21==0 → np.finfo(float).eps to avoid division by zero.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def stot(s_params):
    """Convert 2-port S-parameters to T-parameters.

    Reference: Mavaddat (1996), p.67.
    t = [[1/s21,       -s22/s21],
         [s11/s21, -delta/s21 ]]
    where delta = s11*s22 - s12*s21.

    Input:  (2,2) or (2,2,N) complex array.
    Output: same shape.
    """
    s = np.asarray(s_params, dtype=complex)
    squeeze = s.ndim == 2
    if squeeze:
        s = s[:, :, np.newaxis]          # (2,2,1)

    s11 = s[0, 0, :]
    s12 = s[0, 1, :]
    s21 = s[1, 0, :].copy()
    s22 = s[1, 1, :]

    delta = s11 * s22 - s12 * s21       # MATLAB line 11280
    s21[s21 == 0] = np.finfo(float).eps  # MATLAB line 11281

    t = np.empty_like(s)
    t[0, 0, :] =  1.0 / s21             # MATLAB [1/s21, ...]
    t[0, 1, :] = -s22 / s21
    t[1, 0, :] =  s11 / s21
    t[1, 1, :] = -delta / s21           # MATLAB line 11282

    return t[:, :, 0] if squeeze else t


if __name__ == "__main__":
    import numpy as np
    # Simple 2×2 test: s21=1, others=0 → t = [[1,0],[0,0]]
    s = np.array([[0, 0], [1, 0]], dtype=complex)
    print("stot([[0,0],[1,0]]) =\n", stot(s))
