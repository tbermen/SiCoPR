import numpy as np


def applyDFEbk(hisi, hisi_ref, idx, tap_bk, curval, bmaxg, dfe_delta=0):
    """Apply a contiguous bank of DFE taps at 0-based position idx.

    idx is the 0-based start index (MATLAB used 1-based; callers must subtract 1).
    Returns (hisi_modified, tap_coef, hisi_ref_modified).
    """
    hisi = np.asarray(hisi, dtype=float).copy()
    hisi_ref = np.asarray(hisi_ref, dtype=float).copy()
    sl = slice(idx, idx + tap_bk)
    flt_curval = hisi[sl].copy()

    if dfe_delta != 0:
        flt_curval_q = (
            np.floor(np.abs(flt_curval / curval) / dfe_delta)
            * dfe_delta * np.sign(flt_curval) * curval
        )
    else:
        flt_curval_q = flt_curval

    tap_coef = np.minimum(np.abs(flt_curval_q / curval), bmaxg) * np.sign(flt_curval_q)
    hisi[sl] = hisi[sl] - curval * tap_coef
    hisi_ref[sl] = 0.0
    return hisi, tap_coef, hisi_ref
