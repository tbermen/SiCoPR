# ============================================================
# MATLAB→Python translation notes for force
# MATLAB lines: 6090–6253
# ============================================================
# RxFFE tap solver. Inlines FFE helper.
# Inputs: V (pulse response row), param, OP, ix (0-based cursor), C (taps or []), return_V flag
#
# FFE: for each tap i (0-based, 0..len(C)-1):
#   ishift = (i - cmx) * spui
#   V0 += np.roll(V, ishift) * C[i]
#
# MATLAB vsampled: zero-padded to [zeros(num_taps), vsampled_raw, zeros(cpx)]
# In Python: same 0-based indexing.
# VV matrix: num_taps x num_taps
#   VV[:, i] = vsampled[ivs+i : ivs+i-num_taps+1 : -1]  (reverse stride)
# Solver: VV'\FV' → np.linalg.lstsq(VV.T, FV).
# WIENER_HOPF_MMSE path: undefined in the reference MATLAB -> raises NotImplementedError
#   (not a port gap; the reference itself has no such function). Use FFE_OPT_METHOD='MMSE'.
# Floating taps: uses _findbankloc (inlined).
# RXFFE_FLOAT_CTL: 'taps' uses Cmod for bank, else uses VV row.
# RXFFE_TAP_CONSTRAINT: 'unity cursor' → normalize by Cmod[cmx].
# ============================================================

import numpy as np
from types import SimpleNamespace


def _FFE(C, cmx, spui, V):
    """Inline FFE: apply taps C (length num_taps) to signal V."""
    V = np.asarray(V, dtype=float).ravel()
    C = np.asarray(C, dtype=float).ravel()
    V0 = np.zeros(len(V))
    for i, ci in enumerate(C):
        if ci != 0.0:
            ishift = (i - cmx) * spui
            V0 += np.roll(V, ishift) * ci
    return V0


def _findbankloc(hisi, N_tail_start, N_bmax, N_bf, bmaxg_val, bmaxg, N_bg):
    """Select N_bg non-overlapping banks of size N_bf by highest power."""
    hisi = np.asarray(hisi, dtype=float).ravel()
    n_banks = int(N_bg)
    bank_size = int(N_bf)
    power = np.abs(hisi)
    available = list(range(len(hisi) - bank_size + 1))
    chosen = []
    for _ in range(n_banks):
        if not available:
            break
        best = max(available, key=lambda s: np.sum(power[s:s + bank_size]))
        chosen.extend(range(best, best + bank_size))
        available = [k for k in available if k + bank_size - 1 < best or k > best + bank_size - 1]
    return np.sort(np.array(chosen, dtype=int)) + 1  # 1-based like MATLAB


def force(V, param, OP, ix=None, C=None, return_V=1, chdata=None, txffe=None, Noise_XC=None):
    """RxFFE tap solver (MATLAB lines 6090-6253).

    V: pulse response (1D array).
    ix: 0-based cursor index (default: argmax of V).
    C: pre-computed taps (if not None, skips optimization).
    return_V: if 0, skip computing Vfiltered (speed-up).

    Returns (Vfiltered, Cmod, idx).
    """
    V = np.asarray(V, dtype=float).ravel()

    if ix is None:
        ix = int(np.argmax(V))

    cmx = int(param.RxFFE_cmx)
    cpx = int(param.RxFFE_cpx)

    if int(getattr(param, 'N_bg', 0)) != 0:
        cpx = int(param.N_bmax)

    num_taps = cmx + cpx + 1
    cstep = float(getattr(param, 'RxFFE_stepz', 0))
    ndfe = int(param.ndfe)
    spui = int(param.samples_per_ui)
    param.current_ffegain = 0

    idx = np.array([], dtype=int)

    if return_V and C is not None and len(np.asarray(C)) > 0:
        Vfiltered = _FFE(C, cmx, spui, V)
        return Vfiltered, np.asarray(C, dtype=float).ravel(), idx

    # Build vsampled_raw: samples at spui spacing starting at ix
    N = len(V)
    ix = int(ix)
    mod_ix = ix % spui
    if ix < N:
        if mod_ix == 0:
            pre_start = spui
        else:
            pre_start = mod_ix
        # pre-cursor part
        pre_idx = np.arange(pre_start, ix, spui)
        post_idx = np.arange(ix, N, spui)
        vsampled_raw = np.concatenate([V[pre_idx], V[post_idx]])
    else:
        if mod_ix == 0:
            start = spui
        else:
            start = mod_ix
        vsampled_raw = V[start::spui]

    # Zero-pad: [zeros(num_taps), vsampled_raw, zeros(cpx)]
    vsampled = np.concatenate([np.zeros(num_taps), vsampled_raw, np.zeros(cpx)])

    # Find ivs: index in vsampled matching V[ix]
    if ix < N:
        matches = np.where(vsampled == V[ix])[0]
        ivs = int(matches[0]) if len(matches) > 0 else num_taps
    else:
        ivs = int(np.argmax(vsampled))

    # Build VV matrix (num_taps x num_taps)
    VV = np.zeros((num_taps, num_taps))
    for i in range(num_taps):
        start_idx = ivs + i
        end_idx = start_idx - num_taps + 1
        if end_idx >= 0 and start_idx < len(vsampled):
            col = vsampled[start_idx:end_idx - 1 if end_idx > 0 else None:-1][:num_taps]
            VV[:len(col), i] = col

    if C is None or len(np.asarray(C)) == 0:
        ffe_opt = str(getattr(OP, 'FFE_OPT_METHOD', 'FORCE')).upper()
        if ffe_opt == 'WIENER-HOPF':
            # MATLAB L6188 calls WIENER_HOPF_MMSE(), but that function is NOT defined
            # anywhere in the reference (com_ieee8023_4p14p0.m or any matlab_source.m) —
            # the reference itself errors on this path.  We do not fabricate a solver
            # (would be speculative and, since Wiener-Hopf == MMSE normal equations, a
            # duplicate of the existing 'MMSE' path).  Use FFE_OPT_METHOD='MMSE' instead.
            raise NotImplementedError(
                "force: FFE_OPT_METHOD='WIENER-HOPF' is non-functional in the reference "
                "MATLAB (WIENER_HOPF_MMSE is undefined). Use 'MMSE'.")

        # Build forcing vector FV
        FV = np.zeros(num_taps)
        cursor_gain_dB = float(getattr(param, 'current_ffegain', 0))
        FV[cmx] = vsampled[ivs] * 10 ** (cursor_gain_dB / 20.0)
        if ndfe != 0 and cpx > 0 and ivs + 1 < len(vsampled):
            bmax_arr = np.asarray(param.bmax, dtype=float).ravel()
            bmax_val = float(bmax_arr[0]) if len(bmax_arr) > 0 else 1.0
            FV[cmx + 1] = min(bmax_val * FV[cmx], abs(vsampled[ivs + 1])) * np.sign(vsampled[ivs + 1])

        VVt = VV.T
        if VV.shape[0] == VV.shape[1]:
            try:
                C_solved = np.linalg.solve(VVt, FV)
            except np.linalg.LinAlgError:
                C_solved, _, _, _ = np.linalg.lstsq(VVt, FV, rcond=None)
        else:
            VVVt = VV @ VV.T
            try:
                C_solved = (np.linalg.solve(VVVt, VV) @ FV)
            except np.linalg.LinAlgError:
                C_solved, _, _, _ = np.linalg.lstsq(VV.T, FV, rcond=None)

        Cmod = C_solved[:num_taps]

        # Floating taps
        N_bg = int(getattr(param, 'N_bg', 0))
        if N_bg != 0:
            N_tail_start = int(getattr(param, 'N_tail_start', cpx))
            N_bmax = int(getattr(param, 'N_bmax', cpx))
            N_bf = int(getattr(param, 'N_bf', 1))
            bmaxg = float(getattr(param, 'bmaxg', 1.0))
            float_ctl = str(getattr(OP, 'RXFFE_FLOAT_CTL', 'isi')).lower()
            if float_ctl == 'taps':
                src = Cmod
            else:
                src = VV[cmx, :] if cmx < VV.shape[0] else Cmod
            idx = _findbankloc(src, N_tail_start, N_bmax, N_bf, float(Cmod[cmx]), bmaxg, N_bg)
            idx = np.sort(idx)

        tap_constraint = str(getattr(OP, 'RXFFE_TAP_CONSTRAINT', 'unity cursor')).lower()
        if tap_constraint == 'unity cursor':
            cursor_val = float(Cmod[cmx]) if abs(Cmod[cmx]) > 1e-12 else 1.0
            Cmod = Cmod / cursor_val
        else:
            Cmod = C_solved[:num_taps]

        if cstep != 0:
            Cmod = np.floor(np.abs(Cmod / cstep)) * np.sign(Cmod) * cstep

        if len(idx) > 0:
            N_fixed = cmx + int(getattr(param, 'RxFFE_cpx', cpx)) + 1
            C1 = Cmod.copy()
            C1[N_fixed:] = 0.0
            for j, k in enumerate(idx):
                pos = cmx + 1 + int(k) - 1  # idx is 1-based
                if pos < len(C1):
                    C1[pos] = Cmod[pos] if pos < len(Cmod) else 0.0
            Cmod = C1
    else:
        Cmod = np.asarray(C, dtype=float).ravel()

    if return_V:
        Vfiltered = _FFE(Cmod, cmx, spui, V)
    else:
        Vfiltered = np.array([])

    return Vfiltered, Cmod, idx
