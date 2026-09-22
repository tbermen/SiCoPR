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
    ph_old = np.unwrap(np.angle(S21_old))

    # interp1 sorts its sample points, so a descending (or scrambled) f_old is
    # legal there; PchipInterpolator refused it with "`x` must be strictly
    # increasing sequence".  The unwrap above still runs in the order given.
    # COM Octave, f_old and S21_old both reversed, answers the same values as
    # the ascending call to ~3e-16.
    if f_old.size > 1 and not np.all(np.diff(f_old) > 0):
        order = np.argsort(f_old, kind='stable')
        f_old, mag_db_old, ph_old = f_old[order], mag_db_old[order], ph_old[order]

    mag_db_new = PchipInterpolator(f_old, mag_db_old)(f_new)
    ph_new = PchipInterpolator(f_old, ph_old)(f_new)

    H_new = 10 ** (mag_db_new / 20) * np.exp(1j * ph_new)
    H_new[np.isinf(H_new)] = 0.0

    # MATLAB: inq = find(f_new<=f_b/2,1,'last'); H_new(inq+1:end) = 0.
    # When nothing qualifies inq is [], so []+1:end is an EMPTY index list and
    # the assignment touches nothing.  Zeroing the whole vector here threw the
    # result away.  COM Octave, f_new=[2 7 15 25 35]e9 with f_b=1e6 (and with
    # f_b=0), returns the full interpolated vector, first element
    # 0.97423226153237874-0.11744772561742448j, not zeros.
    below = np.where(f_new <= float(f_b) / 2)[0]
    if below.size:
        inq = int(below[-1])  # last index where f_new <= fb/2
        H_new[inq + 1:] = 0.0

    return H_new
