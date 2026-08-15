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
# Actually MMSE_results.floating_tap_locations=idx+param.RxFFE_cmx+1 in MATLAB means they remain 1-based for reporting. In Python we keep as 0-based (idx + RxFFE_cmx).
# ============================================================

import numpy as np
from scipy.linalg import toeplitz
from types import SimpleNamespace


def _findbankloc(hisi, N_tail_start, N_bmax, N_bf, bmaxg_val, bmaxg, N_bg):
    """Simplified findbankloc: select N_bg*N_bf tap positions with highest power."""
    hisi = np.asarray(hisi, dtype=float).ravel()
    n_banks = int(N_bg)
    bank_size = int(N_bf)
    n_taps = n_banks * bank_size
    power = np.abs(hisi)
    # greedy: pick n_banks non-overlapping banks of size bank_size
    available = list(range(len(hisi) - bank_size + 1))
    chosen = []
    for _ in range(n_banks):
        if not available:
            break
        # find available start with highest bank power
        best = max(available, key=lambda s: np.sum(power[s:s + bank_size]))
        chosen.extend(range(best, best + bank_size))
        # remove overlapping positions
        available = [k for k in available if k + bank_size - 1 < best or k > best + bank_size - 1]
    return np.sort(np.array(chosen, dtype=int)) + 1  # 1-based like MATLAB


def _MMSE_FOM(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx=None):
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
    if len(idx) > 0:
        float_cols = np.asarray(idx, dtype=int) + int(param.RxFFE_cmx)
        col_sel = np.concatenate([np.arange(Nfix), float_cols])
        H = H[:, col_sel]
        Rnn = Rnn[np.ix_(col_sel, col_sel)]

    d = int(d)
    HH = H.T @ H
    R = HH + Rnn / sigma_X2
    Hb = H[d + 1:d + Nb + 1, :]
    h0 = H[d, :]
    ib = np.eye(Nb)
    zb = np.zeros(Nb)
    A = np.block([[R, -Hb.T], [-Hb, ib]])
    C = np.concatenate([h0, zb])
    Ct = C.reshape(-1, 1)
    Z = np.linalg.solve(A, Ct)
    S_inv = float(np.dot(C, Z.ravel()))
    wbl = np.concatenate([Z.ravel(), [1 - S_inv]]) / S_inv
    Nw_used = H.shape[1]
    if len(idx) > 0:
        Nw = Nw_used
    w = wbl[:Nw_used]
    b = wbl[Nw_used:Nw_used + Nb]
    blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b))
    # array_equal, not allclose: MATLAB guards these two branches with ~isequal,
    # which is exact (see com_functions/fn/MMSE_FOM for the full note).
    if Nb > 0 and not np.array_equal(b, blim):
        Rb = np.block([[R, -h0.reshape(-1, 1)], [h0.reshape(1, -1), np.array([[0.0]])]])
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
                sigma_X2, isi_start, isi_end, _MMSE_FOM_fn=_MMSE_FOM)
            idx = np.sort(idx)

    sigma_e, FOM, w, idx_out, Nw_out, blim = _MMSE_FOM(
        param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx)

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
