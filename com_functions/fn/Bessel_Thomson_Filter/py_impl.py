# ============================================================
# MATLAB→Python translation notes for Bessel_Thomson_Filter
# MATLAB lines: 1028–1037
# ============================================================
# bessel(n): called from sibling module — inlined as _bessel (protocol:
#   no cross-module imports from py_impl.py siblings).
# a(1) in MATLAB (1-based) = a[0] in Python = constant term (largest value)
# fliplr(a) reverses [a0,...,an] → [an,...,a0], highest-degree first for polyval
# np.polyval matches MATLAB polyval: highest-degree coefficient first ✓
# use_BT=False: all-pass (unity response)
# ============================================================

import math
import numpy as np


def _factorial(k):
    """MATLAB factorial(): a double, so it overflows to Inf above 170!."""
    return math.inf if k > 170 else math.factorial(k)


def _bessel(n):
    """Bessel polynomial coefficients [a0, a1, ..., an].
    a[k] = coeff of x^k; a[0] = constant term (largest), a[n] = 1.
    """
    # `for ii = 0:n` never runs for n < 0, so MATLAB never assigns `a` and the
    # function errors.  COM Octave: bessel(-1) -> "value on right hand side of
    # assignment is undefined".  Returning an empty array answered a call the
    # reference refuses.
    if n < 0:
        raise ValueError('bessel: output is undefined for n < 0 (got %r)' % (n,))
    # MATLAB factorial() rejects non-integers.  COM Octave: bessel(2.5) ->
    # "factorial: all N must be real non-negative integers".
    if n != int(n):
        raise ValueError('bessel: n must be a non-negative integer (got %r)' % (n,))
    n = int(n)
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        # COM Octave, bessel(90): a(1:10) are Inf.  Python's exact
        # math.factorial made them finite (~1.09e164) instead.
        a[ii] = (_factorial(2 * n - ii)
                 / (2 ** (n - ii) * _factorial(ii) * _factorial(n - ii)))
    return a


def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def Bessel_Thomson_Filter(param, f, use_BT):
    f = np.asarray(f, dtype=float)
    # Same shape as Butterworth_Filter, and the same two traps. MATLAB `if
    # use_BT` is true only for a non-empty value whose elements are ALL
    # non-zero, and `length()` is the LONGEST dimension, not the first.
    # COM Octave (param.BTorder=4, fb=106.25e9, fb_BT_cutoff=0.75):
    #   use_BT=[]    -> ones branch        use_BT=[1 0] -> ones branch
    #   use_BT=[1 1] -> filter branch      f 2x3, use_BT=0 -> ones(1,3)
    #   f scalar, use_BT=0 -> 1            (len(f) raised TypeError)
    use = np.asarray(use_BT)
    if not (use.size and np.all(use)):
        return np.ones(_length(f))
    a = _bessel(param.BTorder)       # [a0,...,an]; a[0] is constant term
    acoef = a[::-1]                  # fliplr → highest-degree first
    s = 1j * f / (param.fb_BT_cutoff * param.fb)
    return a[0] / np.polyval(acoef, s)   # a(1) in MATLAB = a[0]
