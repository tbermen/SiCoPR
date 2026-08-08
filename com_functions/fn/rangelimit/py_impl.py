import copy
import numpy as np


def rangelimit(sch, schFreqAxis, param, OP=None):
    """Limit S-parameter array to param.flim (MATLAB lines 9401-9412)."""
    schFreqAxis = np.asarray(schFreqAxis, dtype=float).ravel()
    sch = np.asarray(sch)
    param_out = copy.copy(param)

    mask = schFreqAxis >= float(param.flim)
    if np.any(mask):
        iend = int(np.argmax(mask)) + 1  # MATLAB sch(1:iend) inclusive
        return sch[:iend], schFreqAxis[:iend], 1, param_out
    param_out.flim = float(schFreqAxis[-1])
    return sch, schFreqAxis, 0, param_out
