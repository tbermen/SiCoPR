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
from com_functions.fn.bessel.py_impl import bessel as _bessel


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
