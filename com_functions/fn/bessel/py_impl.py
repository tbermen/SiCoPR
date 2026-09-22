# ============================================================
# MATLAB→Python translation notes for bessel
# MATLAB lines: 4926–4932
# ============================================================
# 1-based vs 0-based indexing: loop ii=0:n writes to a(ii+1), so
#   MATLAB a[1..n+1] maps to Python a[0..n] — loop index unchanged.
# Column-major vs row-major: output is a 1-D row vector of length n+1.
# MATLAB `end` keyword: not used here.
# Output shape: 1-D array of length n+1 (float64).
# Known discrepancy from prior sicopr.py attempt: none found (simple formula).
# ============================================================
import math
import numpy as np


def _factorial(k):
    """MATLAB factorial(): a double, so it overflows to Inf above 170!."""
    return math.inf if k > 170 else math.factorial(k)


def bessel(n):
    """Return Bessel polynomial coefficients for order n.

    Equivalent to MATLAB bessel(n).
    a[ii] = (2n-ii)! / (2^(n-ii) * ii! * (n-ii)!)  for ii in 0..n
    Returns a 1-D float64 array of length n+1.
    """
    # `for ii = 0:n` never runs for n < 0, so MATLAB never assigns `a` and the
    # function errors.  COM Octave: bessel(-1) -> "value on right hand side of
    # assignment is undefined".  Returning an empty array here answered a call
    # the reference refuses.
    if n < 0:
        raise ValueError('bessel: output is undefined for n < 0 (got %r)' % (n,))
    # MATLAB factorial() rejects non-integers.  COM Octave: bessel(2.5) ->
    # "factorial: all N must be real non-negative integers"; int(n) silently
    # answered bessel(2) instead.
    if n != int(n):
        raise ValueError('bessel: n must be a non-negative integer (got %r)' % (n,))
    n = int(n)
    a = np.empty(n + 1, dtype=float)
    for ii in range(n + 1):  # ii = 0:n in MATLAB
        # COM Octave, bessel(90): a(1:10) are Inf, a(11) = 2.31e157.  Python's
        # exact math.factorial made the first ten finite (~1.09e164) instead.
        a[ii] = _factorial(2 * n - ii) / (
            2 ** (n - ii) * _factorial(ii) * _factorial(n - ii)
        )
    return a


if __name__ == "__main__":
    print("bessel(3) =", bessel(3))
    print("bessel(0) =", bessel(0))
    print("bessel(1) =", bessel(1))
