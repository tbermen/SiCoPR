import copy
import numpy as np
from com_functions.fn.R_series2.py_impl import R_series2 as _R_series2
from com_functions.fn.combines4p.py_impl import combines4p as _combines4p
from com_functions.fn.r_parrelell2.py_impl import r_parrelell2 as _r_parrelell2


def SL(S, f, R, R_0):
    """Add source/load impedance to S-parameters via cascade (MATLAB lines 4368-4403).

    S must have a .Parameters field of shape (2, 2, N) and other sparameters fields.
    Returns a copy SLD with updated S.Parameters.
    """
    SLD = copy.copy(S)
    SLD.Parameters = S.Parameters.copy()
    zref = float(R_0)
    f_arr = np.asarray(f, dtype=float).ravel()

    # R_series2 and r_parrelell2 are called, not re-derived. The private copies
    # that used to live here carried r_parrelell2's *simplified* denominator,
    # -zref/(zref+2*rpad) instead of the reference's -zref/(rpad*(zref/rpad+2)),
    # which is the very rearrangement commit 658bc38 removed from the shared
    # function. COM Octave, r_parrelell2(50, f, rpad):
    #     rpad = -18.75  -> s11 -4.0000000000000009, s21 -3.0000000000000009
    #                       simplified: s11 -4, s21 -3
    #     rpad = 24950   -> s21 0.99899899899899891
    #                       simplified: 0.99899899899899902
    # rpad = -18.75 is SL(R=-30, R_0=50); rpad = 24950 is SL(R=49.9, R_0=50).
    if R == 0:
        return SLD

    if R > zref:
        spr_p = _R_series2(zref, f_arr, float(R) - zref).Parameters
    elif R < zref:
        spr_p = _r_parrelell2(zref, f_arr,
                              -float(R) * zref / (float(R) - zref)).Parameters
    else:
        # MATLAB `SLD=S`, a value copy. Returning the caller's object instead
        # would let a later write to SLD.Parameters reach back into S.
        return SLD

    (SLD.Parameters[0, 0, :], SLD.Parameters[0, 1, :],
     SLD.Parameters[1, 0, :], SLD.Parameters[1, 1, :]) = _combines4p(
        spr_p[0, 0, :], spr_p[0, 1, :], spr_p[1, 0, :], spr_p[1, 1, :],
        S.Parameters[0, 0, :], S.Parameters[0, 1, :], S.Parameters[1, 0, :], S.Parameters[1, 1, :]
    )
    return SLD
