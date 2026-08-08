# ============================================================
# MATLAB→Python translation notes for bessel
# MATLAB lines: 4926–4932
# ============================================================
# 1-based vs 0-based indexing: loop ii=0:n writes to a(ii+1), so
#   MATLAB a[1..n+1] maps to Python a[0..n] — loop index unchanged.
# Column-major vs row-major: output is a 1-D row vector of length n+1.
# MATLAB `end` keyword: not used here.
# Output shape: 1-D array of length n+1 (float64).
# Known discrepancy from prior com.py attempt: none found (simple formula).
# ============================================================
import math
import numpy as np


def bessel(n):
    """Return Bessel polynomial coefficients for order n.

    Equivalent to MATLAB bessel(n).
    a[ii] = (2n-ii)! / (2^(n-ii) * ii! * (n-ii)!)  for ii in 0..n
    Returns a 1-D float64 array of length n+1.
    """
    n = int(n)
    a = np.empty(n + 1, dtype=float)
    for ii in range(n + 1):  # ii = 0:n in MATLAB
        a[ii] = math.factorial(2 * n - ii) / (
            2 ** (n - ii) * math.factorial(ii) * math.factorial(n - ii)
        )
    return a


if __name__ == "__main__":
    print("bessel(3) =", bessel(3))
    print("bessel(0) =", bessel(0))
    print("bessel(1) =", bessel(1))
