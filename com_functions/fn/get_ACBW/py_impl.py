# ============================================================
# MATLAB->Python translation of get_ACBW
# MATLAB lines: 6759-6891 (com_ieee8023_4p17p0.m; new in 4p17p0)
# ============================================================
# Apparent channel bandwidth from the Cumulative Inverse-Channel Penalty:
#   CICP = cumsum(1/|H|^2) * mean(diff f), normalised to its last value, in dB;
#   fit it (get_CICP_fit_sweep), then take the first frequency where the
#   residual's slope reaches T_dev (get_BW_from_CICP_residual).
# Called from FD_Processing only when the workbook sets ACBW = 1 (4p17p0).
#
# Reproduced from the reference as it stands:
#   - The smoothing-window default tests isfield(param,'smooth_window_ghz'),
#     lower case, but sets smooth_window_GHz. The lower-case field never exists,
#     so the window is always 2 GHz, even if smooth_window_GHz was set.
#   - `valid` is computed and never used.
#   - The defaults land on MATLAB's by-value copy of param; the caller's param
#     is not changed, so nothing is written back here either.
# ============================================================

import numpy as np

from com_functions.fn.get_CICP_fit_sweep.py_impl import get_CICP_fit_sweep as _get_CICP_fit_sweep
from com_functions.fn.get_BW_from_CICP_residual.py_impl import get_BW_from_CICP_residual as _get_BW_from_CICP_residual


def _mround(x):
    """MATLAB round(): half away from zero."""
    x = float(x)
    if not np.isfinite(x):
        return x
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not send
    # 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


def get_ACBW(Hch, fGHz, OP, param):
    """Returns (Bch_GHz, CICP_db, CICP_fit_db, CICP_residual, alpha, diff_residual)."""
    def p(name, default):
        return getattr(param, name, default)

    # isfield(param,'smooth_window_ghz') -- the lower-case spelling (see header)
    smooth_window_GHz = p('smooth_window_ghz', None)
    smooth_window_GHz = 2 if smooth_window_GHz is None else param.smooth_window_GHz
    T_dev = p('T_dev', 1)
    f_fit_min_GHz = p('f_fit_min_GHz', 10)
    f_upper_min_GHz = p('f_upper_min_GHz', 20)
    f_upper_max_GHz = p('f_upper_max_GHz', 130)
    f_test_min_GHz = p('f_test_min_GHz', 20)
    f_test_max_GHz = p('f_test_max_GHz', 60)
    step_GHz = p('step_GHz', 5)

    Hch = np.asarray(Hch).squeeze().reshape(-1)
    fGHz = np.asarray(fGHz, dtype=float).squeeze().reshape(-1)
    delta_f = float(np.mean(np.diff(fGHz)))
    Nwin = max(3, _mround(smooth_window_GHz / delta_f))

    salz_density = 1.0 / np.abs(Hch) ** 2
    CICP = np.cumsum(salz_density) * delta_f
    CICP = CICP / CICP[-1]
    with np.errstate(divide='ignore'):
        CICP_db = 20 * np.log10(np.abs(CICP))

    CICP_residual, CICP_fit_db, alpha, idx_fit, idx_valid, f_upper = _get_CICP_fit_sweep(
        CICP_db, fGHz, OP,
        f_fit_min_GHz=f_fit_min_GHz, f_upper_min_GHz=f_upper_min_GHz,
        f_upper_max_GHz=f_upper_max_GHz, f_test_min_GHz=f_test_min_GHz,
        f_test_max_GHz=f_test_max_GHz, step_GHz=step_GHz, Nwin=Nwin)

    Bch_GHz, idx_BW, diff_residual = _get_BW_from_CICP_residual(
        fGHz, CICP_residual, idx_valid, T_dev, OP, smooth_window_GHz=smooth_window_GHz)
    return Bch_GHz, CICP_db, CICP_fit_db, CICP_residual, alpha, diff_residual
