# ============================================================
# MATLAB→Python translation notes for CDF_ev
# MATLAB lines: 1139–1141
# ============================================================
# find(PDF.x >= -val, 1, 'first'): 1-based index of first element ≥ -val.
#   np.argmax(PDF.x >= -val): 0-based index of first True.
#   Both directly index CDF/PDF.x → equivalent result ✓
#   BUT np.argmax returns 0 when *nothing* matches, where find() returns [].
#   See the no-match guard below.
# CDF_ev = CDF(index): scalar probability at the voltage crossing.
#
# Note on the Octave compat build (octave/com_ieee8023_4p16p0_octave_compat.m):
# it replaces the find() with `index = lookup(PDF.x,-val) + 1`. lookup() returns
# the last i with x(i) <= v, so the +1 form answers one bin HIGH exactly when
# -val is bit-identical to a grid element, and runs off the end when -val is
# above the whole axis. Measured under Octave 11.3 on x = -0.3:0.1:0.3 over 13
# probe values: 4 disagree (-val = -0.3, 0.2, 0.3, and the above-range 0.5).
# The other exact-looking hits agree only because 0.1 steps are not exactly
# representable, so which values tie is a floating-point accident rather than
# something a caller can reason about. This port follows the MATLAB reference.
# ============================================================

import numpy as np
from types import SimpleNamespace


def CDF_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    CDF = np.asarray(CDF, dtype=float)
    hit = x >= -val
    if not hit.any():
        # find() is empty, so MATLAB's CDF(index) is an empty 1x0 — there is
        # no value to return.  np.argmax on an all-False mask answers 0, which
        # would hand back CDF(1) as though it were the crossing.
        # COM Octave 4p16p0: CDF_ev(-0.5,PDF,CDF) with PDF.x=[-0.3:0.1:0.3]
        # errors "CDF(8): out of bound 7 (dimensions are 1x7)"; the MATLAB
        # find() form returns empty.  Either way there is no answer here.
        raise IndexError('CDF_ev: no PDF.x >= -val')
    index = int(np.argmax(hit))         # first index where PDF.x >= -val
    return float(CDF[index])


if __name__ == "__main__":
    PDF = SimpleNamespace(x=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]))
    CDF = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0])
    print("CDF_ev(0.15):", CDF_ev(0.15, PDF, CDF))   # 0.35
