# ============================================================
# MATLAB→Python translation notes for R_series2
# MATLAB lines: 4354–4361
# ============================================================
# 1-based vs 0-based: S.Parameters(1,1,:) → params[0,0,:], etc.
# ones(1,length(f))*R → np.full(N, R) or just use scalar R.
# Output struct: SimpleNamespace with Parameters field, shape (2,2,N).
# Formula: series resistor R with reference impedance zref.
#   s11 = s22 = R / (R + 2*zref)
#   s21 = s12 = 2*zref / (R + 2*zref)
# Known discrepancy from prior com.py attempt: none found.
# ============================================================
from types import SimpleNamespace
import numpy as np


def R_series2(zref, f, R):
    """2-port S-parameters for a series resistor R with reference impedance zref.

    Returns SimpleNamespace with Parameters of shape (2, 2, N).
    """
    f = np.asarray(f, dtype=float).ravel()
    N = len(f)

    s11 = R / (R + 2.0 * zref)          # scalar (frequency-independent)
    s21 = (2.0 * zref) / (R + 2.0 * zref)

    params = np.empty((2, 2, N), dtype=complex)
    params[0, 0, :] = s11   # MATLAB S.Parameters(1,1,:)
    params[1, 1, :] = s11   # MATLAB S.Parameters(2,2,:)
    params[1, 0, :] = s21   # MATLAB S.Parameters(2,1,:)
    params[0, 1, :] = s21   # MATLAB S.Parameters(1,2,:)

    S = SimpleNamespace()
    S.Parameters = params
    return S


if __name__ == "__main__":
    import numpy as np
    f = np.array([1e9, 2e9, 3e9])
    S = R_series2(50.0, f, 50.0)
    print("s11:", S.Parameters[0, 0, 0])   # 50/(50+100) = 1/3
    print("s21:", S.Parameters[1, 0, 0])   # 100/150 = 2/3
