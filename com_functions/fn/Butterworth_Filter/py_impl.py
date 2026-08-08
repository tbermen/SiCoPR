# ============================================================
# MATLAB→Python translation notes for Butterworth_Filter
# MATLAB lines: 1133–1138
# ============================================================
# polyval([1 2.613126 3.414214 2.613126 1], s):
#   MATLAB highest-degree-first → np.polyval same convention ✓
# 1i*f./(cutoff*fb): s = j*f/f_c, element-wise → same with numpy arrays
# ones(1,length(f)): row vector → np.ones(len(f)) 1-D array
# use_BW=False: all-pass (unity response)
# ============================================================

import numpy as np

_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]


def Butterworth_Filter(param, f, use_BW):
    f = np.asarray(f, dtype=float)
    if not use_BW:
        return np.ones(len(f))
    s = 1j * f / (param.fb_BW_cutoff * param.fb)
    return 1.0 / np.polyval(_BW_POLY, s)
