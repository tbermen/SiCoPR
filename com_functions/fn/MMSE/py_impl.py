# ============================================================
# MATLAB→Python translation notes for MMSE
# MATLAB lines: 2480–2578
# ============================================================
# samp_idx: MATLAB (mod(cursor_i-1,M)+1):M:length(sbr) → 0-based:
#   start = cursor_i % M (Python 0-based cursor_i)
#   samp_idx = np.arange(start, len(sbr), M)
# dh: find(samp_idx == cursor_i) - 1 → 0-based index in samp_idx where value==cursor_i.
# h: sbr[samp_idx] zero-padded to num_ui.
# H: toeplitz(hc1, hr1) where hc1=[h, zeros(Nw-1)], hr1=[h[0], zeros(Nw-1)].
# Rnn: toeplitz(Rn[0:Nw]).
# RXFFE_FLOAT_CTL=='isi': use findbankloc; else: FOM_rxffe_floating_taps (greedy FOM banks,
#   top-level fn, called with our inlined _MMSE_FOM injected so both use the same kernel).
# MMSE_FOM inlined via import (same package, but per protocol we call it directly).
# Craw = w / w[dw] → normalised by cursor tap.
# floating_tap_locations: MATLAB idx + RxFFE_cmx + 1 (1-based) → Python idx + RxFFE_cmx (0-based? No — MATLAB returns 1-based indices here as locations for reporting).
# MMSE_results.floating_tap_locations = idx + param.RxFFE_cmx + 1 (ML 2576) is
# 1-BASED, and this port matches it -- see the assignment below. An earlier
# version of this comment claimed "in Python we keep as 0-based", which
# contradicted the code directly beneath it and is what made the base of this
# field ambiguous. All three producers now agree on 1-based.
# ============================================================

import numpy as np

def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)

from scipy.linalg import toeplitz
from types import SimpleNamespace



# Nb is fixed for a run; these are rebuilt ~130k times per case otherwise.
_EYE_CACHE = {}
_ZERO_CACHE = {}

def _fb_mask(ndiff, positions, value):
    """ndiff[positions] = value, growing ndiff the way MATLAB would.

    ML 6260/6296 assign `ndiff(new_bank)=min_energy` where new_bank can run
    past the end of ndiff: ndiff is indexed by bank START position, so it is
    tap_bk-1 shorter than h0. MATLAB grows on out-of-range assignment; NumPy
    raises IndexError. The grown entries are -Inf, sort last, and are never
    selected, so the growth is inert -- the divergence was a crash, not a wrong
    answer. See com_functions/fn/findbankloc/py_impl.py.
    """
    positions = np.asarray(positions, dtype=int).ravel()
    if positions.size == 0:
        return ndiff
    need = int(positions.max()) + 1
    if need > ndiff.size:
        ndiff = np.concatenate([ndiff, np.zeros(need - ndiff.size)])
    ndiff[positions] = value
    return ndiff


def _findbankloc(hisi, idx_st, idx_en, tap_bk, curval, bmaxg, N_bg):
    """Faithful port of MATLAB findbankloc (ML 5918-6069).

    This was previously a "pick the highest-power non-overlapping banks"
    approximation, which is not what MATLAB does. The real routine ranks bank
    start positions by ndiff = h0n - h1n and then runs a badV/goodV
    admissibility loop that can reject the strongest bank and pre-commit the
    next one, so the selected set differs from a plain energy ranking.

    idx_st/idx_en are MATLAB 1-based bounds. Returns MATLAB-convention 1-based
    positions in hisi, which is what the callers here expect (MMSE_FOM selects
    columns as idx + RxFFE_cmx; force places taps at cmx + 1 + idx - 1).
    """
    hisi = np.asarray(hisi, dtype=float).ravel()
    idx_st, idx_en = int(idx_st), int(idx_en)
    tap_bk, N_bg = int(tap_bk), int(N_bg)
    len_ = idx_en - idx_st + 1
    h0 = np.abs(hisi[idx_st - 1:idx_en])
    h1 = np.maximum(0.0, h0 - bmaxg * curval)
    if curval < 0:
        # ML 5934: a negative cursor would invert ndiff and make the WEAKEST isi
        # the most desirable, so h1 is forced flat.
        h1 = np.zeros(len_)

    n_bins = len_ - tap_bk + 1
    h0n = np.zeros(n_bins)
    h1n = np.zeros(n_bins)
    for _ii in range(tap_bk):
        h0n += h0[_ii:_ii + n_bins] ** 2
        h1n += h1[_ii:_ii + n_bins] ** 2
    ndiff = h0n - h1n

    def _bad_range(b_start, b_end):
        """Taps closer than one bank below new_bank[0] (ML 6002-6009)."""
        if b_end < 0:
            return np.array([], dtype=int)
        return np.arange(max(0, b_start), b_end + 1, dtype=int)

    MIN_E = -np.inf
    idx = np.full(tap_bk * N_bg, -1, dtype=int)
    ordered_set = np.arange((N_bg - 1) * tap_bk + 1)
    set_next_bank = -1

    for k in range(N_bg):
        # stable, to match MATLAB's sort(...,'descend'); ndiff ties are common
        # because the isi tail is mostly zeros.
        val_sort = np.argsort(-ndiff, kind='stable')

        if k == 0 and np.array_equal(np.sort(val_sort[:len(ordered_set)]), ordered_set):
            idx = np.arange(N_bg * tap_bk)
            break

        if set_next_bank >= 0:
            new_bank = np.arange(set_next_bank, set_next_bank + tap_bk)
            idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
            set_next_bank = -1
            ndiff = _fb_mask(ndiff, new_bank, MIN_E)
            badV = _bad_range(new_bank[0] - tap_bk + 1, new_bank[0] - 1)
            if len(badV):
                ndiff[badV] = MIN_E
            continue

        new_bank = np.arange(val_sort[0], val_sort[0] + tap_bk)
        if k == N_bg - 1:
            idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
            break

        placed = idx[:tap_bk * k]
        badV = np.array([], dtype=int)
        do_it_again, first_time, num_loops = True, True, 0
        while do_it_again:
            do_it_again = False
            if num_loops > len(ndiff):
                break
            badV = _bad_range(new_bank[0] - tap_bk + 1, new_bank[0] - 1)
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

                if (not found_goodV) and len(badV_pos) and _mmin(badV_pos) < ii_found:
                    do_it_again = True
                    ndiff[new_bank[0]] = MIN_E
                    new_bank = np.arange(val_sort[1], val_sort[1] + tap_bk)
                if found_goodV:
                    set_next_bank = goodV_idx
            num_loops += 1

        ndiff = _fb_mask(ndiff, new_bank, MIN_E)
        idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
        if len(badV):
            ndiff[badV] = MIN_E

    # ML 6069 returns idx+idx_st-1 as 1-based; our idx is 0-based within the
    # window, so +idx_st lands on the same 1-based value.
    return idx + idx_st

def _MMSE_FOM(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2,
              idx=None, HH_full=None):
    """Inlined MMSE_FOM for MMSE function."""
    if idx is None or (hasattr(idx, '__len__') and len(idx) == 0):
        idx = np.array([], dtype=int)

    if len(idx) == 0:
        Nw = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        bmax_use = np.asarray(param.bmax, dtype=float).ravel()
        bmin_use = np.asarray(param.bmin, dtype=float).ravel()
    else:
        Nmax = int(param.N_bmax)
        Nfix = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        Nw = int(dw) + Nmax + 1
        bmax_use = np.asarray(param.bmax, dtype=float).ravel()
        bmin_use = np.asarray(param.bmin, dtype=float).ravel()

    Nfix = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
    H = np.asarray(H, dtype=float)
    Rnn = np.asarray(Rnn, dtype=float)
    d = int(d)

    # H is FIXED across the floating-tap bank search - only the column selection
    # changes - yet H.T @ H was recomputed on every call. H is (num_ui+Nw-1, Nw),
    # e.g. 4182x87, so that is ~2.2 MFLOP each time and MMSE_FOM is invoked ~130k
    # times per case. Because (H[:, c].T @ H[:, c]) == (H.T @ H)[ix_(c, c)], the
    # caller can compute the full Gram matrix once and this becomes a small gather.
    # Only the rows the solve actually needs (h0 and Hb) are taken from H itself.
    if HH_full is None:
        HH_full = H.T @ H
    if len(idx) > 0:
        float_cols = np.asarray(idx, dtype=int) + int(param.RxFFE_cmx)
        col_sel = np.concatenate([np.arange(Nfix), float_cols])
        # .take twice beats np.ix_ by ~2.2x for these shapes and is bit-identical;
        # this runs ~130k times per case so the difference is visible.
        HH = HH_full.take(col_sel, 0).take(col_sel, 1)
        Rnn = Rnn.take(col_sel, 0).take(col_sel, 1)
        Hb = H[d + 1:d + Nb + 1, :].take(col_sel, 1)
        h0 = H[d].take(col_sel)
        Nw_cols = len(col_sel)
    else:
        HH = HH_full
        Hb = H[d + 1:d + Nb + 1, :]
        h0 = H[d, :]
        Nw_cols = H.shape[1]

    R = HH + Rnn / sigma_X2
    ib = _EYE_CACHE.get(Nb)
    if ib is None:
        ib = _EYE_CACHE[Nb] = np.eye(Nb)
    zb = _ZERO_CACHE.get(Nb)
    if zb is None:
        zb = _ZERO_CACHE[Nb] = np.zeros(Nb)
    # np.block carries heavy per-call Python overhead and MMSE_FOM is invoked
    # ~130k times per case by the floating-tap bank search. Assembling into a
    # preallocated array is ~3x faster and bit-identical.
    _n = R.shape[0]
    A = np.empty((_n + Nb, _n + Nb), dtype=float)
    A[:_n, :_n] = R
    A[:_n, _n:] = -Hb.T
    A[_n:, :_n] = -Hb
    A[_n:, _n:] = ib
    C = np.concatenate([h0, zb])
    Ct = C.reshape(-1, 1)
    Z = np.linalg.solve(A, Ct)
    S_inv = float(np.dot(C, Z.ravel()))
    wbl = np.concatenate([Z.ravel(), [1 - S_inv]]) / S_inv
    Nw_used = Nw_cols
    if len(idx) > 0:
        Nw = Nw_used
    w = wbl[:Nw_used]
    b = wbl[Nw_used:Nw_used + Nb]
    blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b))
    # array_equal, not allclose: MATLAB guards these two branches with ~isequal,
    # which is exact (see com_functions/fn/MMSE_FOM for the full note).
    if Nb > 0 and not np.array_equal(b, blim):
        _m = R.shape[0]
        Rb = np.empty((_m + 1, _m + 1), dtype=float)
        Rb[:_m, :_m] = R
        Rb[:_m, _m] = -h0
        Rb[_m, :_m] = h0
        Rb[_m, _m] = 0.0
        rhs = np.concatenate([h0 + Hb.T @ blim, [1.0]])
        wl_full = np.linalg.solve(Rb, rhs)
        w = wl_full[:Nw_used]
    wmax_arr = np.asarray(wmax, dtype=float).ravel()[:Nw_used]
    wmin_arr = np.asarray(wmin, dtype=float).ravel()[:Nw_used]
    dw_int = int(dw)
    w_cursor = float(w[dw_int]) if dw_int < len(w) else 1.0
    wlim = np.minimum(wmax_arr * w_cursor, np.maximum(wmin_arr * w_cursor, w))
    # The b/blim refresh belongs INSIDE this branch (MATLAB L2683-2690): with no
    # clipping, blim stays as clip(b) from the solve rather than clip(Hb @ w).
    if not np.array_equal(w, wlim):
        h0w = float(h0 @ wlim)
        if h0w != 0:
            wlim = wlim / h0w
        if Nb > 0:
            b_upd = Hb @ wlim
            blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b_upd))
    w = wlim
    Hb_T_blim = Hb.T @ blim if Nb > 0 else np.zeros_like(h0)
    sigma_e = float(np.sqrt(np.maximum(0.0, sigma_X2 * (
        float(w @ R @ w) + 1.0 + float(np.dot(blim, blim))
        - 2.0 * float(np.dot(w, h0)) - 2.0 * float(np.dot(w, Hb_T_blim))
    ))))
    R_LM = float(param.R_LM)
    L = int(param.levels)
    FOM = float(20.0 * np.log10(R_LM / (L - 1) / sigma_e)) if sigma_e > 0 else np.inf
    return sigma_e, FOM, w, idx, Nw_used, blim


def MMSE(PSD_results, sbr, cursor_i, param, OP):
    """MMSE RxFFE optimisation (MATLAB lines 2480-2578).

    cursor_i: 0-based Python index.
    Returns MMSE_results SimpleNamespace.
    """
    num_ui = int(param.num_ui_RXFF_noise)
    M = int(param.samples_per_ui)
    L = int(param.levels)
    sigma_X2 = float((L ** 2 - 1) / (3 * (L - 1) ** 2))
    fb = float(param.fb)
    cursor_i = int(cursor_i)

    # MATLAB: samp_idx = (mod(cursor_i-1,M)+1):M:length(sbr) (1-based)
    # Python (0-based): start = cursor_i % M
    sbr = np.asarray(sbr, dtype=float).ravel()
    start = cursor_i % M
    samp_idx = np.arange(start, len(sbr), M)  # 0-based

    # dh: index in samp_idx where value == cursor_i
    matches = np.where(samp_idx == cursor_i)[0]
    dh = int(matches[0]) if len(matches) > 0 else 0

    dw = int(param.RxFFE_cmx)
    h = sbr[samp_idx].ravel()
    if len(h) < num_ui:
        h = np.concatenate([h, np.zeros(num_ui - len(h))])
    h = h[:num_ui]
    N = len(h)

    Nb = int(param.ndfe)
    d = dw + dh

    S_n = np.asarray(PSD_results.S_n, dtype=float).ravel()
    Rn = np.real(np.fft.ifft(S_n)) * fb

    if int(param.N_bg) == 0:
        Nw = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        bmax = np.asarray(param.bmax, dtype=float).ravel()
        bmin = np.asarray(param.bmin, dtype=float).ravel()
        wmax = np.concatenate([
            np.ones(int(param.RxFFE_cmx) - 1) * float(param.ffe_tapn_max),
            [float(param.ffe_pre_tap1_max)],
            [1.0],
            [float(param.ffe_post_tap1_max)],
            np.ones(int(param.RxFFE_cpx) - 1) * float(param.ffe_tapn_max),
        ])
        wmin = -wmax.copy()
        wmin[int(param.RxFFE_cmx)] = 1.0
        wmax[int(param.RxFFE_cmx)] = 1.0
        idx = np.array([], dtype=int)
    else:
        Nfloating_taps = int(param.N_bf) * int(param.N_bg)
        Nmax = int(param.N_bmax)
        Nfix = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        Nw = dw + Nmax + 1
        bmax = np.asarray(param.bmax, dtype=float).ravel()
        bmin = np.asarray(param.bmin, dtype=float).ravel()
        bmaxg = float(param.bmaxg)
        wmax = np.concatenate([
            np.ones(int(param.RxFFE_cmx) - 1) * float(param.ffe_tapn_max),
            [float(param.ffe_pre_tap1_max)], [1.0], [float(param.ffe_post_tap1_max)],
            np.ones(int(param.RxFFE_cpx) - 1) * float(param.ffe_tapn_max),
            np.ones(Nfloating_taps) * bmaxg,
        ])
        wmin = np.concatenate([
            -np.ones(int(param.RxFFE_cmx) - 1) * float(param.ffe_tapn_max),
            [-float(param.ffe_pre_tap1_max)], [1.0], [-float(param.ffe_post_tap1_max)],
            -np.ones(int(param.RxFFE_cpx) - 1) * float(param.ffe_tapn_max),
            -np.ones(Nfloating_taps) * bmaxg,
        ])
        idx = np.array([], dtype=int)

    isi_start = dh + 1  # 0-based start of ISI region
    isi_end = (dh - dw) + Nw  # 0-based end

    if isi_end > len(h):
        raise ValueError(f'num_ui_RXFF_noise ({num_ui}) is too small')

    hc1 = np.concatenate([h, np.zeros(Nw - 1)])
    hr1 = np.concatenate([[h[0]], np.zeros(Nw - 1)])
    if len(samp_idx) < num_ui and len(h) > len(samp_idx) + Nw:
        H = toeplitz(h[:len(samp_idx) + Nw - 1], hr1)
    else:
        H = toeplitz(hc1, hr1)
    Rnn = toeplitz(Rn[:Nw], Rn[:Nw])
    # Gram matrix of the (large, fixed) H, computed once and reused by every
    # MMSE_FOM evaluation below - see the note in _MMSE_FOM.
    HH_full = H.T @ H

    if int(param.N_bg) != 0:
        ctl = str(getattr(OP, 'RXFFE_FLOAT_CTL', 'isi')).lower()
        if ctl == 'isi':
            hisi = h[isi_start:isi_end]
            idx = _findbankloc(hisi, int(param.RxFFE_cpx) + 1, int(param.N_bmax),
                               int(param.N_bf), np.inf, float(param.bmaxg), int(param.N_bg))
            idx = np.sort(idx)
        else:
            idx = FOM_rxffe_floating_taps(
                param, h, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax,
                sigma_X2, isi_start, isi_end, _MMSE_FOM_fn=_MMSE_FOM,
                HH_full=HH_full)
            idx = np.sort(idx)

    sigma_e, FOM, w, idx_out, Nw_out, blim = _MMSE_FOM(
        param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
        HH_full=HH_full)

    Craw = w / (w[dw] if abs(w[dw]) > 1e-12 else 1.0)

    if int(param.N_bg) != 0:
        Nfix = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        # MATLAB:
        #   C = Craw;
        #   C(Nfix+1 : Nmax+param.ffe_pre_tap_len+1) = 0;
        #   C(idx+param.RxFFE_cmx+1) = Craw(Nfix+(1:Nfloating_taps));
        #
        # Both MATLAB assignments AUTO-EXTEND C with zeros, so the returned filter
        # spans the floating-tap positions (which reach far past the fixed taps —
        # e.g. index 72 for a 23-element Craw). Craw holds the tap *count*
        # (Nfix + Nfloating), not the tap *span*, so C must be grown explicitly
        # here. Previously C was left at len(Craw) and every floating tap landing
        # beyond it was silently discarded, truncating the equalizer and leaving
        # sigma_e 2.5-4.5% low (FOM high by 0.2-0.4 dB, worse on lossier packages).
        idx_arr = np.asarray(idx_out, dtype=int).ravel()
        n_end = int(param.N_bmax) + int(param.ffe_pre_tap_len) + 1
        span = max(len(Craw), n_end)
        if idx_arr.size:
            span = max(span, int(idx_arr.max()) + int(param.RxFFE_cmx) + 1)
        C = np.zeros(span, dtype=float)
        C[:len(Craw)] = Craw
        C[Nfix:n_end] = 0.0
        for j, k in enumerate(idx_arr):
            c_col = int(k) + int(param.RxFFE_cmx)
            C[c_col] = Craw[Nfix + j] if Nfix + j < len(Craw) else 0.0
    else:
        C = Craw

    r = SimpleNamespace()
    r.sigma_e = sigma_e
    r.FOM = FOM
    r.C = C
    r.floating_tap_locations = idx_out + int(param.RxFFE_cmx) + 1  # 1-based for reporting
    r.blim = blim
    r.Nw = Nw_out
    return r
