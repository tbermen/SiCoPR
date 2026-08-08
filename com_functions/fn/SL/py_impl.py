import copy
import numpy as np


def _R_series2_params(zref, f, R):
    N = len(f)
    s = complex(R / (R + 2 * zref))
    t = complex(2 * zref / (R + 2 * zref))
    p = np.empty((2, 2, N), dtype=complex)
    p[0, 0, :] = p[1, 1, :] = s
    p[1, 0, :] = p[0, 1, :] = t
    return p


def _r_parrelell2_params(zref, f, rpad):
    N = len(f)
    d = zref + 2 * rpad
    p = np.empty((2, 2, N), dtype=complex)
    p[0, 0, :] = p[1, 1, :] = complex(-zref / d)
    p[1, 0, :] = p[0, 1, :] = complex(2 * rpad / d)
    return p


def _combines4p(a11, a12, a21, a22, b11, b12, b21, b22):
    def sq(x): return np.asarray(x, dtype=complex).ravel()
    a11, a12, a21, a22 = sq(a11), sq(a12), sq(a21), sq(a22)
    b11, b12, b21, b22 = sq(b11), sq(b12), sq(b21), sq(b22)
    Nv = 1 - a22 * b11
    return (a11 + a12 * a21 * b11 / Nv, a12 * b12 / Nv,
            b21 * a21 / Nv, b22 + b12 * b21 * a22 / Nv)


def SL(S, f, R, R_0):
    """Add source/load impedance to S-parameters via cascade (MATLAB lines 4368-4403).

    S must have a .Parameters field of shape (2, 2, N) and other sparameters fields.
    Returns a copy SLD with updated S.Parameters.
    """
    SLD = copy.copy(S)
    SLD.Parameters = S.Parameters.copy()
    zref = float(R_0)
    f_arr = np.asarray(f, dtype=float).ravel()

    if R == 0:
        return S

    if R > zref:
        spr_p = _R_series2_params(zref, f_arr, float(R) - zref)
    elif R < zref:
        spr_p = _r_parrelell2_params(zref, f_arr, -float(R) * zref / (float(R) - zref))
    else:
        return S

    (SLD.Parameters[0, 0, :], SLD.Parameters[0, 1, :],
     SLD.Parameters[1, 0, :], SLD.Parameters[1, 1, :]) = _combines4p(
        spr_p[0, 0, :], spr_p[0, 1, :], spr_p[1, 0, :], spr_p[1, 1, :],
        S.Parameters[0, 0, :], S.Parameters[0, 1, :], S.Parameters[1, 0, :], S.Parameters[1, 1, :]
    )
    return SLD
