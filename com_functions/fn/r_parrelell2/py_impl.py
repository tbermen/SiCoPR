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

    # MATLAB L9391/9393 written out exactly:
    #     S(1,1,:) = -zref/(rpad*(zref/rpad + 2))
    #     S(2,1,:) =  2/(zref/rpad + 2)
    # The algebraically equivalent -zref/(zref+2*rpad) and 2*rpad/(zref+2*rpad)
    # are NOT equivalent in floating point. They differ in the last bits for
    # most finite rpad (5 of 9 sampled values), and diverge outright at the
    # limits. Verified against COM Octave:
    #     rpad=Inf -> S11 -0 (negative zero), S21 1     simplified: S21 NaN
    #     rpad=0   -> S11 NaN, S21 0                    simplified: S11 -1
    # rpad=Inf is the natural "no pad resistor" setting, so the simplified form
    # put a NaN straight into the package cascade.
    # np.float64 rather than Python floats: MATLAB's zref/0 is Inf, while
    # Python's raises ZeroDivisionError. errstate keeps the Inf and NaN quiet,
    # as MATLAB produces them without a warning.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.float64(zref) / np.float64(rpad)
        s11 = -np.float64(zref) / (np.float64(rpad) * (ratio + 2.0))
        s21 = 2.0 / (ratio + 2.0)

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
