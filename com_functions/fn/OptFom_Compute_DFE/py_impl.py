# ============================================================
# MATLAB→Python translation notes for OptFom_Compute_DFE
# MATLAB lines: 3219–3304
# ============================================================
# cursor_i is 0-based in Python (MATLAB 1-based → Python 0-based).
# dfecursors: MATLAB sbr(cursor_i+M*(1):M:cursor_i+M*(ndfe)) (1-based)
#   → Python sbr[cursor_i+M : cursor_i+M*ndfe+1 : M] (0-based)
# postcursors for floatingDFE: MATLAB sbr(cursor_i+M:M:end) (1-based)
#   → Python sbr[cursor_i+M :: M] (0-based)
# dfecursors_windowed: MATLAB sbr(cursor_i-T_O+M*(1):M:cursor_i+M*ndfe-T_O) (1-based)
#   → Python sbr[cursor_i-T_O+M : cursor_i+M*ndfe-T_O+1 : M] (0-based)
# N_tail_start: 1-based in MATLAB config → subtract 1 for 0-based array access.
# dfe_clipper and floatingDFE inlined as _dfe_clipper and _floatingDFE.
# ============================================================

import numpy as np


def _dfe_clipper(input_arr, max_threshold, min_threshold):
    inp = np.asarray(input_arr, dtype=float)
    hi = np.asarray(max_threshold, dtype=float)
    lo = np.asarray(min_threshold, dtype=float)
    is_row = inp.ndim <= 1 or (inp.ndim == 2 and inp.shape[0] == 1)
    if is_row:
        hi = hi.ravel()
        lo = lo.ravel()
    else:
        hi = hi.ravel().reshape(-1, 1)
        lo = lo.ravel().reshape(-1, 1)
    out = inp.copy()
    out[inp > hi] = hi[inp > hi]
    out[inp < lo] = lo[inp < lo]
    return out


def _findbankloc(hisi, idx_st, idx_en, tap_bk, curval, bmaxg, N_bg):
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
        val_sort = np.argsort(-ndiff)
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
            badV = np.arange(max(0, b_start), b_end + 1, dtype=int) if b_end >= 0 else np.array([], dtype=int)
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
            badV = np.arange(max(0, b_start), b_end + 1, dtype=int) if b_end >= 0 else np.array([], dtype=int)
            if len(badV) and len(placed):
                badV = badV[~np.isin(badV, placed)]
            goodV_idx = new_bank[0] - tap_bk
            if len(badV) > 0:
                if not first_time:
                    val_sort = np.argsort(-ndiff)
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
    return idx + (idx_st - 1)


def _floatingDFE(hisi, N_b, N_bf, N_bg, N_bmax, bmaxg, curval, dfe_delta=0):
    hisi = np.asarray(hisi, dtype=float).copy()
    tap_coef = np.zeros(len(hisi))
    b = np.zeros(len(hisi))
    tap_loc = _findbankloc(hisi, N_b + 1, N_bmax, N_bf, curval, bmaxg, N_bg)
    flt_curval = hisi[tap_loc].copy()
    if dfe_delta != 0:
        flt_curval_q = (np.floor(np.abs(flt_curval / curval) / dfe_delta)
                        * dfe_delta * np.sign(flt_curval) * curval)
    else:
        flt_curval_q = flt_curval
    applied_coef = np.minimum(np.abs(flt_curval_q / curval), bmaxg) * np.sign(flt_curval_q)
    hisi[tap_loc] -= curval * applied_coef
    tap_coef[tap_loc] = applied_coef
    tap_loc = np.sort(tap_loc)
    b[tap_loc] = bmaxg
    return tap_loc, tap_coef, hisi, b


def OptFom_Compute_DFE(sbr, THIS, param, do_C2M, T_O):
    sbr = np.asarray(sbr, dtype=float)
    cursor_i = int(THIS.cursor_i)  # 0-based
    M = int(param.samples_per_ui)
    ndfe = int(param.ndfe)
    N_tail_start = int(param.N_tail_start)

    # Equation 93A-27: DFE cursor samples (0-based Python)
    dfecursors = sbr[cursor_i + M : cursor_i + M * ndfe + 1 : M]

    if param.dfe_delta != 0:
        dfecursors_q = (np.floor(np.abs(dfecursors / sbr[cursor_i]) / param.dfe_delta)
                        * param.dfe_delta * np.sign(dfecursors) * sbr[cursor_i])
    else:
        dfecursors_q = dfecursors.copy()

    if param.Floating_DFE:
        postcursors = sbr[cursor_i + M :: M]
        floating_tap_locations, floating_tap_coef, hisi, bmax = _floatingDFE(
            postcursors, param.ndfe_passed, param.N_bf, param.N_bg,
            param.N_bmax, param.bmaxg, sbr[cursor_i], param.dfe_delta)
        bmax_arr = np.asarray(bmax)
        newbmax = np.concatenate([np.asarray(param.bmax).ravel(),
                                   bmax_arr[param.ndfe_passed:param.N_bmax]])
        param.use_bmax = newbmax.reshape(-1, 1) if hasattr(param, 'use_bmax') and np.asarray(param.use_bmax).ndim > 1 else newbmax
        param.use_bmin = np.concatenate([np.asarray(param.bmin).ravel(),
                                          bmax_arr[param.ndfe_passed:param.N_bmax] * -1]).reshape(param.use_bmax.shape) if hasattr(param, 'use_bmax') else np.concatenate([np.asarray(param.bmin).ravel(), bmax_arr[param.ndfe_passed:param.N_bmax] * -1])
    else:
        param.use_bmax = param.bmax
        param.use_bmin = param.bmin
        floating_tap_coef = []

    actual_dfecursors = _dfe_clipper(
        dfecursors_q,
        sbr[cursor_i] * np.asarray(param.use_bmax).ravel(),
        sbr[cursor_i] * np.asarray(param.use_bmin).ravel())

    if do_C2M:
        dfecursors_windowed = sbr[cursor_i - T_O + M : cursor_i + M * ndfe - T_O + 1 : M]
        excess_dfe_cursors = dfecursors_windowed - actual_dfecursors
    else:
        excess_dfe_cursors = dfecursors - actual_dfecursors

    dfetaps = actual_dfecursors / sbr[cursor_i]

    if len(dfetaps) >= N_tail_start and N_tail_start != 0:
        tail_taps = dfetaps[N_tail_start - 1:]  # 1-based → 0-based
        tail_RSS = float(np.linalg.norm(tail_taps))
        if tail_RSS != 0:
            if tail_RSS >= param.B_float_RSS_MAX:
                scale = min(tail_RSS, param.B_float_RSS_MAX) / tail_RSS
                use_bmax = np.asarray(param.use_bmax).ravel().copy()
                use_bmin = np.asarray(param.use_bmin).ravel().copy()
                use_bmax[N_tail_start - 1:] = scale * np.abs(tail_taps)
                use_bmin[N_tail_start - 1:] = -scale * np.abs(tail_taps)
                param.use_bmax = use_bmax
                param.use_bmin = use_bmin

                actual_dfecursors = _dfe_clipper(
                    dfecursors_q,
                    sbr[cursor_i] * np.asarray(param.use_bmax).ravel(),
                    sbr[cursor_i] * np.asarray(param.use_bmin).ravel())
                if do_C2M:
                    excess_dfe_cursors = dfecursors_windowed - actual_dfecursors
                else:
                    excess_dfe_cursors = dfecursors - actual_dfecursors
                dfetaps = actual_dfecursors / sbr[cursor_i]
    else:
        tail_RSS = 0.0

    THIS.dfetaps = dfetaps
    if param.Floating_DFE:
        THIS.floating_tap_locations = floating_tap_locations
    THIS.floating_tap_coef = floating_tap_coef
    THIS.tail_RSS = tail_RSS
    THIS.excess_dfe_cursors = excess_dfe_cursors
    return THIS, param
