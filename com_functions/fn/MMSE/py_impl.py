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
# The canonical MMSE_FOM, not a copy: an inlined copy here stopped tracking the
# original (4p17p0 changed MMSE_FOM). Behaviourally identical to the copy it replaced.
from com_functions.fn.MMSE_FOM.py_impl import MMSE_FOM as _MMSE_FOM



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
    # 4p17p0 L2584: HH_val = transpose(H(:,1))*H, formed once here and passed to
    # MMSE_FOM, which builds HH from it by lag instead of H'*H per candidate.
    # Equal to H'*H up to summation order here, since the truncated branch above
    # drops only h's zero padding (see MMSE_FOM). Earlier releases form H'*H, so
    # HH_val stays None for them.
    HH_val = None
    if str(getattr(param, 'matlab_version', '4p15p0')) >= '4p17p0':
        HH_val = H[:, 0] @ H
    Rnn = toeplitz(Rn[:Nw], Rn[:Nw])

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
                HH_unique_values=HH_val)
            idx = np.sort(idx)

    sigma_e, FOM, w, idx_out, Nw_out, blim = _MMSE_FOM(
        param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
        HH_val=HH_val)

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
