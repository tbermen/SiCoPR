import numpy as np
from com_functions.fn.findbankloc.py_impl import findbankloc as _findbankloc

def floatingDFE(hisi, N_b, N_bf, N_bg, N_bmax, bmaxg, curval, dfe_delta=0):
    """Find and apply N_bg groups of N_bf floating DFE taps.

    N_b: number of fixed taps (floating taps start after these, 0-based: position N_b).
    N_bmax: maximum tap index (1-based MATLAB convention; internally converted).
    Returns (tap_loc, tap_coef, hisi_modified, b) all using 0-based indices.
    """
    hisi = np.asarray(hisi, dtype=float).copy()
    tap_coef = np.zeros(len(hisi))
    b = np.zeros(len(hisi))

    # MATLAB: findbankloc(hisi, N_b+1, N_bmax, N_bf, curval, bmaxg, N_bg)
    # idx_st=N_b+1 (1-based), idx_en=N_bmax (1-based)
    tap_loc = _findbankloc(hisi, N_b + 1, N_bmax, N_bf, curval, bmaxg, N_bg)

    flt_curval = hisi[tap_loc].copy()
    if dfe_delta != 0:
        flt_curval_q = (
            np.floor(np.abs(flt_curval / curval) / dfe_delta)
            * dfe_delta * np.sign(flt_curval) * curval
        )
    else:
        flt_curval_q = flt_curval

    applied_coef = np.minimum(np.abs(flt_curval_q / curval), bmaxg) * np.sign(flt_curval_q)
    hisi[tap_loc] -= curval * applied_coef
    tap_coef[tap_loc] = applied_coef
    tap_loc = np.sort(tap_loc)
    b[tap_loc] = bmaxg
    return tap_loc, tap_coef, hisi, b
