import numpy as np
from com_functions.fn.s21_to_impulse_DC.py_impl import s21_to_impulse_DC


def get_RAW_FIR(H, f, OP, param):
    """Apply Butterworth H_r then convert to impulse response. MATLAB lines 6686-6692."""
    f = np.asarray(f, dtype=float).ravel()
    H = np.asarray(H, dtype=complex).ravel()
    H_r = 1.0 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1],
                            1j * f / (0.75 * float(param.fb)))
    H_filtered = H * H_r
    FIR, t, _, _ = s21_to_impulse_DC(H_filtered, f, float(param.sample_dt), OP, param)
    return FIR, t
