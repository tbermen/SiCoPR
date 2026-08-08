import numpy as np
from scipy.interpolate import PchipInterpolator


def H_interp(S21_old, f_old, f_new, f_b):
    """Interpolate S21 to a new frequency grid using pchip (MATLAB lines 2180-2198).

    Magnitude is interpolated in dB; phase is unwrapped then interpolated.
    Points above f_b/2 are zeroed to prevent extrapolation blowup.
    """
    S21_old = np.asarray(S21_old, dtype=complex).ravel()
    f_old = np.asarray(f_old, dtype=float).ravel()
    f_new = np.asarray(f_new, dtype=float).ravel()

    mag_db_old = 20 * np.log10(np.abs(S21_old))
    mag_db_new = PchipInterpolator(f_old, mag_db_old)(f_new)

    ph_old = np.unwrap(np.angle(S21_old))
    ph_new = PchipInterpolator(f_old, ph_old)(f_new)

    H_new = 10 ** (mag_db_new / 20) * np.exp(1j * ph_new)
    H_new[np.isinf(H_new)] = 0.0

    below = f_new <= float(f_b) / 2
    if np.any(below):
        inq = int(np.where(below)[0][-1])  # last index where f_new <= fb/2
        H_new[inq + 1:] = 0.0
    else:
        H_new[:] = 0.0

    return H_new
