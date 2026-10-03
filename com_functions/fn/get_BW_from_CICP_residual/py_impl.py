# ============================================================
# MATLAB->Python translation of get_BW_from_CICP_residual
# MATLAB lines: 6893-6995 (com_ieee8023_4p17p0.m; new in 4p17p0)
# ============================================================
# Bandwidth = first frequency inside idx_fit where the slope of the smoothed
# CICP residual reaches T_dev (dB/GHz); the last frequency if none does.
#
# Two MATLAB built-ins are reproduced, not borrowed from numpy:
#   movmean(x, N, 'omitnan')  N odd: (N-1)/2 each side; N even: N/2 before and
#                             N/2-1 after; the window shrinks at the ends.
#   gradient(y, x)            centred difference over the two neighbours,
#                             one-sided at the ends. numpy.gradient with an x
#                             array uses a second-order non-uniform formula,
#                             which is not what MATLAB computes.
# Index: idx_BW is returned 0-based (MATLAB's find(...,1,'first') minus one).
# ============================================================

import numpy as np


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


def _movmean_omitnan(x, N):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if N % 2:
        kb = kf = (N - 1) // 2
    else:
        kb, kf = N // 2, N // 2 - 1
    out = np.empty(n)
    for i in range(n):
        seg = x[max(0, i - kb):min(n, i + kf + 1)]
        seg = seg[~np.isnan(seg)]
        out[i] = seg.mean() if seg.size else np.nan
    return out


def _gradient(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    g = np.empty(n)
    if n == 1:
        g[0] = 0.0
        return g
    g[0] = (y[1] - y[0]) / (x[1] - x[0])
    g[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    if n > 2:
        g[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    return g


def _row(x):
    return np.asarray(x).squeeze().reshape(-1)


def get_BW_from_CICP_residual(fGHz, residual_db, idx_fit, T_dev, OP,
                              smooth_window_GHz=None):
    """Returns (Bch_GHz, idx_BW, diff_residual). idx_BW is 0-based.

    smooth_window_GHz stands for the 'smooth_window_ghz' name-value pair; MATLAB
    has no default for it and fails if it is not given, as this does.
    """
    fGHz = _row(fGHz).astype(float)
    residual_db = _row(residual_db).astype(float)
    idx_fit = _row(idx_fit)
    if len(fGHz) != len(residual_db):
        raise ValueError('fGHz and residual_db must have the same length.')
    if len(idx_fit) != len(fGHz):
        raise ValueError('idx_fit must have the same length as fGHz.')
    valid = np.isfinite(fGHz) & np.isfinite(residual_db)
    idx_fit = idx_fit.astype(bool) & valid
    if np.count_nonzero(idx_fit) < 3:
        raise ValueError('idx_fit does not contain enough valid samples.')
    df_GHz = float(np.median(np.diff(fGHz[valid])))
    if not np.isfinite(df_GHz) or df_GHz <= 0:
        raise ValueError('Invalid frequency axis.')
    if smooth_window_GHz is None:
        # MATLAB: 'smooth_window_GHz' undefined -> error at first use
        raise ValueError("get_BW_from_CICP_residual: 'smooth_window_ghz' was not given")

    Nwin = int(max(3, _mround(smooth_window_GHz / df_GHz)))
    res_smooth = _movmean_omitnan(residual_db, Nwin)
    dDdf = _gradient(res_smooth, fGHz)
    masked = dDdf * idx_fit                   # NaN * 0 stays NaN, as in MATLAB
    hits = np.nonzero(masked >= T_dev)[0]
    if hits.size == 0:
        idx_BW = len(fGHz) - 1
        if getattr(OP, 'DISPLAY_WINDOW', 0):
            print('No strong rolloff detected: reporting max frequency')
    else:
        idx_BW = int(hits[0])
    diff_residual = masked
    Bch_GHz = float(fGHz[idx_BW])
    return Bch_GHz, idx_BW, diff_residual
