import numpy as np


def _findbankloc(hisi, idx_st, idx_en, tap_bk, curval, bmaxg, N_bg):
    """Inlined findbankloc. Returns 0-based indices into hisi."""
    hisi = np.asarray(hisi, dtype=float).ravel()
    len_ = idx_en - idx_st + 1
    h0 = np.abs(hisi[idx_st - 1:idx_en])
    h1 = np.maximum(0.0, h0 - bmaxg * curval)
    if curval < 0:
        h1 = np.zeros(len_)
    n_bins = len_ - tap_bk + 1
    h0n = np.zeros(n_bins)
    h1n = np.zeros(n_bins)
    for ii in range(tap_bk):
        h0n += h0[ii:ii + n_bins] ** 2
        h1n += h1[ii:ii + n_bins] ** 2
    ndiff = h0n - h1n
    MIN_E = -np.inf
    idx = np.full(tap_bk * N_bg, -1, dtype=int)
    ordered_set = np.arange((N_bg - 1) * tap_bk + 1)
    set_next_bank = -1

    for k in range(N_bg):
        val_sort = np.argsort(-ndiff, kind='stable')
        if k == 0:
            ns = len(ordered_set)
            if np.array_equal(np.sort(val_sort[:ns]), ordered_set):
                idx = np.arange(N_bg * tap_bk)
                break
        if set_next_bank >= 0:
            new_bank = np.arange(set_next_bank, set_next_bank + tap_bk)
            idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
            set_next_bank = -1
            ndiff[new_bank] = MIN_E
            b_start = new_bank[0] - tap_bk + 1
            b_end = new_bank[0] - 1
            badV = _bv(b_start, b_end)
            if len(badV):
                ndiff[badV] = MIN_E
            continue
        new_bank = np.arange(val_sort[0], val_sort[0] + tap_bk)
        if k == N_bg - 1:
            idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
            break
        placed = idx[:tap_bk * k]
        badV = np.array([], dtype=int)
        do_it_again = True
        first_time = True
        num_loops = 0
        while do_it_again:
            do_it_again = False
            if num_loops > len(ndiff):
                break
            b_start = new_bank[0] - tap_bk + 1
            b_end = new_bank[0] - 1
            badV = _bv(b_start, b_end)
            if len(badV) and len(placed):
                badV = badV[~np.isin(badV, placed)]
            goodV_idx = new_bank[0] - tap_bk
            if len(badV) > 0:
                if not first_time:
                    val_sort = np.argsort(-ndiff, kind='stable')
                first_time = False
                checkV = np.concatenate([badV, new_bank])
                badV_pos = np.array([int(np.where(val_sort == v)[0][0]) for v in badV])
                found_goodV = False
                ii_found = len(val_sort) - 1
                for ii_vs in range(len(val_sort)):
                    if val_sort[ii_vs] == goodV_idx:
                        found_goodV = True
                        ii_found = ii_vs
                        break
                    if not np.any(val_sort[ii_vs] == checkV):
                        ii_found = ii_vs
                        break
                if (not found_goodV) and len(badV_pos) > 0 and np.min(badV_pos) < ii_found:
                    do_it_again = True
                    ndiff[new_bank[0]] = MIN_E
                    new_bank = np.arange(val_sort[1], val_sort[1] + tap_bk)
                if found_goodV:
                    set_next_bank = goodV_idx
            num_loops += 1
        ndiff[new_bank] = MIN_E
        idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
        if len(badV):
            ndiff[badV] = MIN_E

    return idx + (idx_st - 1)   # 0-based hisi positions


def _bv(b_start, b_end):
    if b_end < 0:
        return np.array([], dtype=int)
    return np.arange(max(0, b_start), b_end + 1, dtype=int)


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
