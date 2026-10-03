# ============================================================
# MATLAB->Python translation of get_CICP_fit_sweep
# MATLAB lines: 7077-7222 (com_ieee8023_4p17p0.m; new in 4p17p0)
# ============================================================
# Fits CICP_db over [f_fit_min, f_upper] for f_upper = f_upper_min:step:f_upper_max,
# scores each fit by the slope of its residual over [f_test_min, f_test_max]
# (polyfit order 1), and keeps the flattest: the FIRST minimum of |slope|.
#
# Reproduced from the reference as it stands:
#   - the default block assigns f_test_min_GHz twice (20, then 60) and never
#     defaults f_test_max_GHz; get_ACBW always passes both, and a call without
#     f_test_max_ghz fails here as it does in MATLAB.
#   - smoothed_residual, rms_residual_smoothed, idx_ref and fGHz_fit are computed
#     and never used, and nothing they call can fail, so they are not computed.
#     Nwin is accepted for the same reason and has no effect on any output.
#   - the diagnostic figure is behind `OP.DISPLAY_WINDOW && 0`: never drawn.
# polyfit(x,y,1) is done as MATLAB and Octave do it, [Q,R]=qr(V,0); p=R\(Q'*y),
# not by numpy.polyfit (an SVD least-squares solve).
# ============================================================

import numpy as np
from scipy.linalg import solve_triangular

from com_functions.fn.get_CICP_fit_residual.py_impl import get_CICP_fit_residual as _get_CICP_fit_residual


def _colon(a, step, b):
    """MATLAB a:step:b."""
    n = int(np.floor((b - a) / step + 1e-10))
    return [a + k * step for k in range(n + 1)] if n >= 0 else []


def _polyfit1(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        # MATLAB polyfit on fewer points than coefficients warns and returns a
        # rank-deficient fit; with no points at all it errors. The range here
        # is empty only when fGHz never reaches f_test_max_GHz.
        raise ValueError('get_CICP_fit_sweep: polyfit range %d points; fGHz must reach '
                         'f_test_max_GHz' % x.size)
    V = np.column_stack([x, np.ones_like(x)])
    Q, R = np.linalg.qr(V, mode='reduced')
    # R\(Q'*y): MATLAB sees R is upper triangular and back-substitutes
    return solve_triangular(R, Q.T @ y, lower=False)


def _first_index(mask):
    hits = np.nonzero(mask)[0]
    return int(hits[0]) if hits.size else None


def get_CICP_fit_sweep(CICP_db, fGHz, OP, f_fit_min_GHz=5, f_upper_min_GHz=10,
                       f_upper_max_GHz=130, f_test_min_GHz=60, f_test_max_GHz=None,
                       step_GHz=10, Nwin=3):
    """Returns (CICP_residual, CICP_fit_db, alpha, idx_fit, idx_valid, f_upper).

    Keyword defaults are the reference's, including f_test_min_GHz = 60 (its
    second assignment wins) and no default for f_test_max_GHz.
    """
    CICP_db = np.asarray(CICP_db, dtype=float).squeeze().reshape(-1)
    fGHz = np.asarray(fGHz, dtype=float).squeeze().reshape(-1)
    if f_test_max_GHz is None:
        raise ValueError("get_CICP_fit_sweep: 'f_test_max_GHz' undefined")

    fits, fom, f_upper_list = [], [], []
    for f_upper_tmp in _colon(f_upper_min_GHz, step_GHz, f_upper_max_GHz):
        mask = (fGHz >= f_fit_min_GHz) & (fGHz <= f_upper_tmp)
        res_ext, fit_db, alpha, mask = _get_CICP_fit_residual(CICP_db, fGHz, mask)
        ixmax = _first_index(fGHz >= f_test_max_GHz)
        ixmin = _first_index(fGHz >= f_test_min_GHz)
        if ixmax is None or ixmin is None:
            sl = slice(0, 0)        # MATLAB ixmin:[] is empty
        else:
            sl = slice(ixmin, ixmax + 1)
        p = _polyfit1(fGHz[sl], res_ext[sl])
        fits.append((res_ext, fit_db, alpha, mask))
        fom.append(float(p[0]))
        f_upper_list.append(f_upper_tmp)

    afom = np.abs(np.asarray(fom))
    if afom.size == 0 or np.all(np.isnan(afom)):
        raise ValueError('get_CICP_fit_sweep: no candidate fit')
    best = int(np.nonzero(afom == np.nanmin(afom))[0][0])
    f_upper = f_upper_list[best]
    CICP_residual, CICP_fit_db, alpha, idx_fit = fits[best]
    idx_valid = fGHz >= f_fit_min_GHz
    return CICP_residual, CICP_fit_db, alpha, idx_fit, idx_valid, f_upper
