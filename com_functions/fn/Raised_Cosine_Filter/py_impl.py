# ============================================================
# MATLAB→Python translation notes for Raised_Cosine_Filter
# MATLAB lines: 4362–4367
# ============================================================
# Calls Tukey_Window with explicit fr=param.RC_Start, fb=param.RC_end.
# The canonical Tukey_Window is imported rather than inlined: the copy that
#   used to live here was an element-wise np.where, which the oracle pass over
#   Tukey_Window had already proved wrong (MATLAB concatenates three counted
#   pieces), and a private copy is exactly how that fix failed to arrive.
# use_RC=False: all-pass (unity response)
# ============================================================

import numpy as np
from com_functions.fn.Tukey_Window.py_impl import Tukey_Window as _tukey_window


def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def Raised_Cosine_Filter(param, f, use_RC):
    f = np.asarray(f, dtype=float)
    # Third of the family, and it carries the same two traps as
    # Butterworth_Filter and Bessel_Thomson_Filter. MATLAB `if use_RC` is true
    # only for a non-empty value whose elements are ALL non-zero.
    # COM Octave (RC_Start=20e9, RC_end=40e9):
    #   use_RC=[]    -> ones branch        use_RC=[1 0] -> ones branch
    #   use_RC=[1 1] -> Tukey branch       use_RC=2     -> Tukey branch
    use = np.asarray(use_RC)
    if not (use.size and np.all(use)):
        # ones(1,length(f)): length() is the LONGEST dimension, not the first.
        # COM Octave: f 3x5 -> ones(1,5), five ones not three; f scalar -> 1
        # (len(f) raised TypeError on a scalar).
        return np.ones(_length(f))
    return _tukey_window(f, param, param.RC_Start, param.RC_end)
