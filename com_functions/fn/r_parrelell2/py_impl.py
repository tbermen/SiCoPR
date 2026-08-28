# ============================================================
# MATLAB→Python translation notes for r_parrelell2
# MATLAB lines: 9390–9400
# ============================================================
# 1-based vs 0-based: Parameters(1,1,:) → params[0,0,:], etc.
# Simplified formulas (factor out rpad from denominator):
#   s11 = s22 = -zref / (rpad*(zref/rpad + 2)) = -zref / (zref + 2*rpad)
#   s21 = s12 = 2 / (zref/rpad + 2) = 2*rpad / (zref + 2*rpad)
# Output struct: SimpleNamespace with Parameters of shape (2,2,N).
# Frequency-independent (values broadcast across all N freq points).
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
from types import SimpleNamespace
import numpy as np


def r_parrelell2(zref, f, rpad):
    """2-port S-parameters for a shunt (parallel) resistor rpad.

    Returns SimpleNamespace with Parameters of shape (2, 2, N).
    """
    f = np.asarray(f, dtype=float).ravel()
    N = len(f)

    denom = zref + 2.0 * rpad
    s11 = -zref / denom                  # MATLAB line 9391
    s21 = 2.0 * rpad / denom             # MATLAB line 9393

    params = np.empty((2, 2, N), dtype=complex)
    params[0, 0, :] = s11
    params[1, 1, :] = s11
    params[1, 0, :] = s21
    params[0, 1, :] = s21

    S = SimpleNamespace()
    S.Parameters = params
    return S


if __name__ == "__main__":
    import numpy as np
    f = np.array([1e9])
    S = r_parrelell2(50.0, f, 50.0)
    # denom=150, s11=-50/150=-1/3, s21=100/150=2/3
    print("s11:", S.Parameters[0, 0, 0], "  s21:", S.Parameters[1, 0, 0])
