import numpy as np


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
        BEST.PSD_results = THIS.PSD_results
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
