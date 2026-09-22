# ============================================================
# MATLAB→Python translation notes for Butterworth_Filter
# MATLAB lines: 1133–1138
# ============================================================
# polyval([1 2.613126 3.414214 2.613126 1], s):
#   MATLAB highest-degree-first → np.polyval same convention ✓
# 1i*f./(cutoff*fb): s = j*f/f_c, element-wise → same with numpy arrays
# ones(1,length(f)): row vector → np.ones(_length(f)) 1-D array, where
#   _length() is MATLAB's length(): the longest dimension, 1 for a scalar
# use_BW=False: all-pass (unity response)
# ============================================================

import numpy as np

_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]


def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def Butterworth_Filter(param, f, use_BW):
    f = np.asarray(f, dtype=float)
    # MATLAB `if use_BW` is true only for a non-empty value whose elements are
    # ALL non-zero.  COM Octave: use_BW=[] -> ones branch, [1 0] -> ones branch,
    # [1 1] -> filter branch.  `not use_BW` raised on any numpy array of more
    # than one element, and took the filter branch for the list [1, 0].
    use = np.asarray(use_BW)
    if not (use.size and np.all(use)):
        # ones(1,length(f)): length() is the LONGEST dimension, not the first.
        # COM Octave: f 2x3 -> ones(1,3), three elements not six; f scalar -> 1
        # (len(f) raised TypeError on a scalar).
        return np.ones(_length(f))
    s = 1j * f / (param.fb_BW_cutoff * param.fb)
    return 1.0 / np.polyval(_BW_POLY, s)
