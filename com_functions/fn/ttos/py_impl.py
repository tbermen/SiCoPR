# ============================================================
# MATLAB→Python translation notes for ttos
# MATLAB lines: 11318–11324
# ============================================================
# 1-based vs 0-based: t_params(1,1,:) → t[0,0,:]; etc.
# Input/output shape: (2,2) or (2,2,N) — same as stot.
# eps guard: t11==0 → np.finfo(float).eps.
# Formula: s = [[t21/t11, delta/t11], [1/t11, -t12/t11]]
#   where delta = t11*t22 - t21*t12.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def ttos(t_params):
    """Convert 2-port T-parameters to S-parameters.

    Reference: Mavaddat (1996), p.67.
    s = [[t21/t11, delta/t11],
         [1/t11,  -t12/t11 ]]
    where delta = t11*t22 - t21*t12.

    Input:  (2,2) or (2,2,N) complex array.
    Output: same shape.
    """
    t = np.asarray(t_params, dtype=complex)
    squeeze = t.ndim == 2
    if squeeze:
        t = t[:, :, np.newaxis]

    t11 = t[0, 0, :].copy()
    t12 = t[0, 1, :]
    t21 = t[1, 0, :]
    t22 = t[1, 1, :]

    delta = t11 * t22 - t21 * t12       # MATLAB line 11322
    t11[t11 == 0] = np.finfo(float).eps  # MATLAB line 11323

    s = np.empty_like(t)
    s[0, 0, :] =  t21 / t11             # MATLAB line 11324
    s[0, 1, :] =  delta / t11
    s[1, 0, :] =  1.0 / t11
    s[1, 1, :] = -t12 / t11

    return s[:, :, 0] if squeeze else s


if __name__ == "__main__":
    import numpy as np
    t = np.array([[1+0j, 0+0j], [0+0j, 0+0j]])
    print("ttos([[1,0],[0,0]]) =\n", ttos(t))
