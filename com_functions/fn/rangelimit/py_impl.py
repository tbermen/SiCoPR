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
        # MATLAB indexes sch(1:iend,:,:), so an axis longer than sch's first
        # dimension is an out-of-bound READ, not a short slice.  COM Octave,
        # sch=1:6 (a 1x6 row) with a 6-point axis and flim=3e9:
        #   "error: sch(4,_,_): out of bound 1 (dimensions are 1x6)".
        # numpy returned the whole row and reported limited=1 on it.
        nrow = sch.shape[0] if sch.ndim else 1
        if iend > nrow:
            raise IndexError(
                'rangelimit: sch(%d,_,_): out of bound %d (dimensions are %s)'
                % (iend, nrow,
                   'x'.join(str(n) for n in (sch.shape or (1, 1)))))
        return sch[:iend], schFreqAxis[:iend], 1, param_out
    param_out.flim = float(schFreqAxis[-1])
    return sch, schFreqAxis, 0, param_out
