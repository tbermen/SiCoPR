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


def _bessel(n):
    """Bessel polynomial coefficients [a0, a1, ..., an].
    a[k] = coeff of x^k; a[0] = constant term (largest), a[n] = 1.
    """
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        a[ii] = (math.factorial(2 * n - ii)
                 / (2 ** (n - ii) * math.factorial(ii) * math.factorial(n - ii)))
    return a


def Bessel_Thomson_Filter(param, f, use_BT):
    f = np.asarray(f, dtype=float)
    if not use_BT:
        return np.ones(len(f))
    a = _bessel(param.BTorder)       # [a0,...,an]; a[0] is constant term
    acoef = a[::-1]                  # fliplr → highest-degree first
    s = 1j * f / (param.fb_BT_cutoff * param.fb)
    return a[0] / np.polyval(acoef, s)   # a(1) in MATLAB = a[0]
