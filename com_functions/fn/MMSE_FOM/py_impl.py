# ============================================================
# MATLAB→Python translation notes for MMSE_FOM
# MATLAB lines: 2580–2692
# ============================================================
# Uses block-matrix inversion (speedup path) to solve for RxFFE taps w and DFE taps b.
# toeplitz(Rnn) built externally and passed in; H already built externally.
# idx: empty (no floating taps) or 1-based MATLAB indices → 0-based Python: idx-1.
# H: when N_bg!=0, columns are subset [1:Nfix, idx+RxFFE_cmx+1] (1-based) →
#   Python: H[:, list(range(Nfix)) + list(idx + RxFFE_cmx)].
# Rnn: similarly subsetted.
# d: dw + dh (0-based in the passed value).
# Hb = H[d+1:d+Nb+1, :]  (MATLAB d+2:d+Nb+1 1-based → Python d+1:d+Nb+1 0-based).
# h0 = H[d, :]  (MATLAB d+1 1-based → Python d 0-based).
# Block solve: A=[R -Hb'; -Hb I], C=[h0 zb], wbl = ([Z;1-S_inv]/S_inv).
# blim = clip(b, bmin, bmax).
# wlim = clip(w, wmin*w[dw], wmax*w[dw]).
# If w != wlim: wlim = wlim / (h0 @ wlim).
# sigma_e = sqrt(sigma_X2 * (w'Rw + 1 + b'b - 2*w'h0' - 2*w'Hb'b)).
# FOM = 20*log10(R_LM / (L-1) / sigma_e).
# ============================================================

import numpy as np
from scipy.linalg import toeplitz



# Nb is fixed for a run; these are rebuilt ~130k times per case otherwise.
_EYE_CACHE = {}
_ZERO_CACHE = {}

def MMSE_FOM(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2,
             idx=None):
    """Compute MMSE FOM and optimal equalizer taps (MATLAB lines 2580-2692).

    Returns (sigma_e, FOM, w, idx, Nw, blim).
    idx: None or 0-based integer array of floating tap positions.
    """
    if idx is None or (hasattr(idx, '__len__') and len(idx) == 0):
        idx = np.array([], dtype=int)

    if len(idx) == 0:
        Nw = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        bmax_use = np.asarray(param.bmax, dtype=float).ravel()
        bmin_use = np.asarray(param.bmin, dtype=float).ravel()
    else:
        Nfloating_taps = len(idx)
        Nmax = int(param.N_bmax)
        Nfix = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)
        Nw = int(dw) + Nmax + 1
        bmax_use = np.asarray(param.bmax, dtype=float).ravel()
        bmin_use = np.asarray(param.bmin, dtype=float).ravel()

    Nfix = int(param.RxFFE_cmx) + 1 + int(param.RxFFE_cpx)

    H = np.asarray(H, dtype=float)
    Rnn = np.asarray(Rnn, dtype=float)
    d = int(d)
    # ML 2609-2619: H = H(:, [1:Nfix idx+cmx+1]); HH = H'*H; then Hb and h0 are
    # rows of the SELECTED H. The Gram matrix is formed per call from the
    # selected columns, as the reference forms it. (It was once hoisted out of
    # the floating-tap search as (H'*H)(sel,sel) of the full H, for speed; that
    # sums in a different order, so it was removed on 2026-09-24 with the other
    # speed-ups that had never been verified against the reference.)
    if len(idx) > 0:
        float_cols = (np.asarray(idx, dtype=int) + int(param.RxFFE_cmx))  # 0-based cols
        col_sel = np.concatenate([np.arange(Nfix), float_cols])
        # .take is a gather, bit-identical to np.ix_ indexing
        Hs = H.take(col_sel, 1)
        HH = Hs.T @ Hs
        Rnn = Rnn.take(col_sel, 0).take(col_sel, 1)
        Hb = Hs[d + 1:d + Nb + 1, :]
        h0 = Hs[d]
        Nw_cols = len(col_sel)
    else:
        HH = H.T @ H
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

    # Block matrix solve (speedup path)
    # Preallocated assembly rather than np.block: this is a hot path
    # (~130k calls per case from the floating-tap bank search) and np.block's
    # per-call overhead dominates. Bit-identical result.
    _n = R.shape[0]
    A = np.empty((_n + Nb, _n + Nb), dtype=float)
    A[:_n, :_n] = R
    A[:_n, _n:] = -Hb.T
    A[_n:, :_n] = -Hb
    A[_n:, _n:] = ib
    C = np.concatenate([h0, zb])    # row vector as 1D
    Ct = C.reshape(-1, 1)           # column vector
    # ML 2645: Z = A\\Ct on a SQUARE system. MATLAB warns and returns Inf
    # when A is exactly singular, and the NaNs that follow make this candidate
    # lose; numpy raises. Measured under Octave with MATLAB backslash
    # semantics: sigma_e NaN, FOM NaN, w all NaN. Following the 2026-09-23
    # force() ruling the port stops rather than absorbing the case, but it
    # says which solve failed instead of reporting a bare LinAlgError.
    try:
        Z = np.linalg.solve(A, Ct)
    except np.linalg.LinAlgError:
        raise ValueError(
            'MMSE_FOM: the [R -Hb\'; -Hb ib] system is singular to working '
            'precision, so the tap solve has no unique answer. MATLAB returns '
            'Inf here and the candidate loses; SiCoPR stops instead, so the '
            'degenerate case is visible. Nw=%d, Nb=%d.' % (Nw_cols, Nb))
    S_inv = float(np.dot(C, Z.ravel()))
    wbl = np.concatenate([Z.ravel(), [1 - S_inv]]) / S_inv

    Nw_used = Nw_cols
    if len(idx) > 0:
        Nw = Nw_used  # re-adjust Nw to number of used taps

    w = wbl[:Nw_used]
    b = wbl[Nw_used:Nw_used + Nb]

    # Apply blim
    blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b))
    # MATLAB guards with ~isequal(...), which is an exact comparison. np.allclose
    # carries a 1e-5 relative tolerance and can take the opposite branch on taps
    # that were clipped by a tiny amount.
    if Nb > 0 and not np.array_equal(b, blim):
        _m = R.shape[0]
        Rb = np.empty((_m + 1, _m + 1), dtype=float)
        Rb[:_m, :_m] = R
        Rb[:_m, _m] = -h0
        Rb[_m, :_m] = h0
        Rb[_m, _m] = 0.0
        rhs = np.concatenate([h0 + Hb.T @ blim, [1.0]])
        # ML 2669, the same square backslash after the DFE taps are clipped
        try:
            wl_full = np.linalg.solve(Rb, rhs)
        except np.linalg.LinAlgError:
            raise ValueError(
                'MMSE_FOM: the clipped-DFE system [R -h0; h0 0] is singular '
                'to working precision. MATLAB returns Inf here and the '
                'candidate loses; SiCoPR stops instead. Nw=%d, Nb=%d.'
                % (Nw_cols, Nb))
        w = wl_full[:Nw_used]

    # Apply wlim
    wmax_arr = np.asarray(wmax, dtype=float).ravel()
    wmin_arr = np.asarray(wmin, dtype=float).ravel()
    if len(wmax_arr) > Nw_used:
        wmax_arr = wmax_arr[:Nw_used]
        wmin_arr = wmin_arr[:Nw_used]
    dw_int = int(dw)
    w_cursor = float(w[dw_int]) if dw_int < len(w) else 1.0
    wlim = np.minimum(wmax_arr * w_cursor, np.maximum(wmin_arr * w_cursor, w))
    # MATLAB:
    #   if ~isequal(w, wlim)
    #       wlim = wlim/(h0*wlim);
    #       if Nb > 0
    #           b = Hb*wlim; blim = min(bmax, max(bmin, b));
    #       end
    #   end
    #   w = wlim; b = blim;
    #
    # The b/blim refresh is INSIDE the clipping branch. When no tap was clipped,
    # MATLAB keeps blim = clip(b) from the MMSE solve above; recomputing it here
    # as clip(Hb @ w) substitutes a different vector, and blim feeds sigma_e
    # directly (the b'b and -2 w'Hb'b terms), biasing FOM.
    if not np.array_equal(w, wlim):
        h0w = float(h0 @ wlim)
        if h0w != 0:
            wlim = wlim / h0w
        if Nb > 0:
            b_upd = Hb @ wlim
            blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b_upd))

    w = wlim

    # sigma_e
    Hb_T_blim = Hb.T @ blim if Nb > 0 else np.zeros_like(h0)
    sigma_e = float(np.sqrt(np.maximum(0.0, sigma_X2 * (
        float(w @ R @ w)
        + 1.0
        + float(np.dot(blim, blim))
        - 2.0 * float(np.dot(w, h0))
        - 2.0 * float(np.dot(w, Hb_T_blim))
    ))))

    R_LM = float(param.R_LM)
    L = int(param.levels)
    FOM = float(20.0 * np.log10(R_LM / (L - 1) / sigma_e)) if sigma_e > 0 else np.inf

    return sigma_e, FOM, w, idx, Nw_used, blim
