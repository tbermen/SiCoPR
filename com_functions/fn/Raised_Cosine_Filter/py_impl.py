# ============================================================
# MATLAB→Python translation notes for Raised_Cosine_Filter
# MATLAB lines: 4362–4367
# ============================================================
# Calls Tukey_Window with explicit fr=param.RC_Start, fb=param.RC_end.
# Protocol: no cross-module imports from sibling py_impl.py files →
#   _tukey_window inlined here (matches Tukey_Window/py_impl.py exactly).
# use_RC=False: all-pass (unity response)
# ============================================================

import numpy as np


def _tukey_window(f, fr, fb):
    fperiod = 2 * (fb - fr)
    H_tw = np.where(
        f < fr,
        1.0,
        np.where(
            (f >= fr) & (f <= fb),
            0.5 * np.cos(2 * np.pi * (f - fb) / fperiod - np.pi) + 0.5,
            0.0,
        ),
    )
    return H_tw[:len(f)]


def Raised_Cosine_Filter(param, f, use_RC):
    f = np.asarray(f, dtype=float)
    if not use_RC:
        return np.ones(len(f))
    return _tukey_window(f, param.RC_Start, param.RC_end)
