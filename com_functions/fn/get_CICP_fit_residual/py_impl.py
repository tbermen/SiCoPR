# ============================================================
# MATLAB->Python translation of get_CICP_fit_residual
# MATLAB lines: 6997-7075 (com_ieee8023_4p17p0.m; new in 4p17p0)
# ============================================================
# COM-style basis fit of CICP_db over a logical mask:
#   efit_db = a0 + a1*sqrt(f) + a2*f + a3*f^2, extrapolated over all of fGHz,
#   residual_db = CICP_db - efit_db.
# alpha = (fmbg'*fmbg) \ (fmbg'*LGw): a SQUARE symmetric system. MATLAB's and
# Octave's backslash try a Cholesky factorisation first when the matrix is
# symmetric with a positive diagonal, and fall back to LU only if that fails,
# so the solve is done the same way here rather than by np.linalg.solve (LU).
# The fit weights w are all ones, so w .* x is x exactly.
# ============================================================

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _backslash_sym(A, b):
    """MATLAB A\\b for a symmetric A with a positive diagonal: Cholesky, else LU.

    On an exactly singular system MATLAB warns and returns Inf, and the fit and
    everything after it is Inf or NaN. As with the other square solves in the
    port (the 2026-09-23 force() ruling), SiCoPR stops instead and names the
    solve, rather than substituting a least-squares answer the reference never
    gives. Reached only by a fit window whose samples are degenerate.
    """
    try:
        return cho_solve(cho_factor(A, lower=False, check_finite=False), b,
                         check_finite=False)
    except np.linalg.LinAlgError:
        pass
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError('get_CICP_fit_residual: the 4x4 CICP fit system is singular '
                         '(degenerate fit window); MATLAB returns Inf here, SiCoPR stops')


def _row(x):
    return np.asarray(x).squeeze().reshape(-1)


def get_CICP_fit_residual(CICP_db, fGHz, idx_fit):
    """Returns (residual_db, efit_db, alpha, idx_fit), all row vectors.

    idx_fit comes back cleaned: the mask actually used (finite, f >= 0).
    """
    CICP_db = _row(CICP_db).astype(float)
    fGHz = _row(fGHz).astype(float)
    idx_fit = _row(idx_fit)
    if len(CICP_db) != len(fGHz):
        raise ValueError('CICP_db and fGHz must have the same length.')
    if len(idx_fit) != len(fGHz):
        raise ValueError('idx_fit must have the same length as fGHz.')
    valid = np.isfinite(CICP_db) & np.isfinite(fGHz) & (fGHz >= 0)
    idx_fit = idx_fit.astype(bool) & valid
    if np.count_nonzero(idx_fit) < 4:
        raise ValueError('Fit region must contain at least 4 valid samples.')

    x = fGHz[idx_fit]
    y = CICP_db[idx_fit]
    fmbg = np.column_stack([np.ones(len(x)), np.sqrt(x), x, x ** 2])
    LGw = y
    alpha = _backslash_sym(fmbg.T @ fmbg, fmbg.T @ LGw)

    efit_db = alpha[0] + alpha[1] * np.sqrt(fGHz) + alpha[2] * fGHz + alpha[3] * fGHz ** 2
    residual_db = CICP_db - efit_db
    return residual_db, efit_db, alpha, idx_fit
