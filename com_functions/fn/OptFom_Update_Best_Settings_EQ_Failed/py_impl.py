import numpy as np
from types import SimpleNamespace

def _value_copy(obj):
    """MATLAB assigns structs BY VALUE; Python binds a reference.

    get_PSDs mutates its `result` argument in place, and optimize_fom reuses one
    PSD_results object for every txffe/itick candidate inside a CTLE block. A
    plain `BEST.PSD_results = THIS.PSD_results` therefore leaves BEST pointing at
    an object that keeps being overwritten, so it ends up holding the LAST tick
    swept rather than the winning one. The COM stage never rebuilds S_tn/S_jn/
    S_rj_jn/S_xn -- it copies them out of fom_result.PSD_results and only rescales
    by |H_rxffe|^2 (ML 547-553) -- so the stale arrays land straight in the
    reported noise terms. Detach on store, which is what MATLAB does.
    """
    if obj is None:
        return None
    out = SimpleNamespace(**vars(obj))
    for k, v in vars(out).items():
        if isinstance(v, np.ndarray):
            setattr(out, k, v.copy())
        elif isinstance(v, list):
            setattr(out, k, [
                SimpleNamespace(**{kk: (vv.copy() if isinstance(vv, np.ndarray) else vv)
                                   for kk, vv in vars(e).items()})
                if hasattr(e, '__dict__') else e
                for e in v])
    return out



# --- inline from FFE (MATLAB 2026-2048) ---
def _FFE(C, cmx, spui, V):
    C = np.asarray(C, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 2 and V.shape[1] == 1:
        V = V.ravel()
    V0 = 0.0
    for i, c in enumerate(C):
        if c != 0:
            ishift = (i - cmx) * spui
            V0 = np.roll(V, ishift) * c + V0
    return V0


def OptFom_Update_Best_Settings_EQ_Failed(BEST, THIS, sbr, chdata, param, OP):
    """Update BEST when EQ optimization fails (MATLAB lines 3891-3934).

    Returns BEST.
    """
    sbr = np.asarray(sbr, dtype=float).ravel()
    BEST.bmax = param.bmax
    BEST.bmin = param.bmin
    BEST.tail_RSS = 0.0
    BEST.ffegain = 0.0
    BEST.txffe = THIS.txffe
    BEST.sbr = sbr
    BEST.ctle = THIS.ctle_index

    if OP.RxFFE:
        BEST.PSD_results = _value_copy(THIS.PSD_results)
        BEST.MMSE_results = THIS.MMSE_results
        BEST.RxFFE = THIS.C

    BEST.G_high_pass = THIS.g_LP_index
    BEST.FOM = THIS.FOM

    cursor_i = THIS.cursor_i
    # If cursor_i is None or empty → use argmax of sbr
    if cursor_i is None or (hasattr(cursor_i, '__len__') and len(cursor_i) == 0):
        cursor_i = int(np.argmax(sbr))
    BEST.cursor_i = cursor_i
    BEST.itick = THIS.itick

    if not OP.TDMODE:
        M = int(param.samples_per_ui)
        cmx = int(param.cursor_index) - 1  # 0-based cursor
        if hasattr(chdata[0], 'ctle_imp_response'):
            BEST.IR = _FFE(THIS.txffe, cmx, M, chdata[0].ctle_imp_response)
        else:
            BEST.IR = []

    BEST.sigma_N = THIS.sigma_N
    BEST.h_J = THIS.h_J
    BEST.A_p = float(np.max(sbr))
    BEST.ISI = 1.0

    # DFE taps: UI-spaced postcursors 1..ndfe, normalized by cursor
    M = int(param.samples_per_ui)
    ndfe = int(param.ndfe)
    ci = int(cursor_i)
    cursor_val = float(sbr[ci])
    # MATLAB: sbr(cursor_i+M:M:cursor_i+M*ndfe)/sbr(cursor_i) — both 1-based
    # Python: sbr[ci+M : ci+M*ndfe+1 : M] / cursor_val (0-based ci)
    BEST.dfetaps = sbr[ci + M: ci + M * ndfe + 1: M] / cursor_val
    BEST.A_s = cursor_val

    if param.Floating_DFE:
        BEST.floating_tap_locations = []
        BEST.floating_tap_coef = []

    return BEST
