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


def OptFom_Update_Best_Setttings(BEST, THIS, sbr, chdata, param, OP):
    """Copy THIS/param fields into BEST after a successful FOM update (MATLAB lines 3935-3975).

    Note: function name preserves original MATLAB typo (three t's).
    Returns BEST.
    """
    BEST.ffegain = param.current_ffegain
    BEST.txffe = THIS.txffe
    BEST.txffe_index = THIS.tx_index_vector
    BEST.sbr = sbr
    BEST.ctle = THIS.ctle_index
    BEST.gdc = THIS.g_dc
    BEST.G_high_pass = THIS.g_LP_index
    BEST.FOM = THIS.FOM
    BEST.cursor_i = THIS.cursor_i
    BEST.itick = THIS.itick

    if not OP.TDMODE:
        M = int(param.samples_per_ui)
        cmx = int(param.cursor_index) - 1  # 0-based cursor index
        BEST.IR = _FFE(THIS.txffe, cmx, M, chdata[0].ctle_imp_response)

    # Stash the genuine CTLE frequency response computed for this (winning)
    # setting so the engineering .mat export can reproduce the FD equalizer
    # chain without recomputing it. H_ctf = CTLE transfer fn; sdd21ctf =
    # channel cascaded with CTLE (both on chdata[0].faxis). No effect on COM.
    BEST.H_ctf = getattr(THIS, 'H_ctf', None)
    BEST.sdd21ctf = getattr(chdata[0], 'sdd21ctf', None)

    BEST.sigma_N = THIS.sigma_N
    BEST.h_J = THIS.h_J
    BEST.A_s = THIS.A_s
    BEST.A_p = THIS.A_p
    BEST.ISI = THIS.ISI_N
    BEST.bmax = param.use_bmax
    BEST.bmin = param.use_bmin
    BEST.tail_RSS = THIS.tail_RSS
    BEST.dfetaps = THIS.dfetaps

    if param.Floating_DFE:
        BEST.floating_tap_locations = THIS.floating_tap_locations
        BEST.floating_tap_coef = THIS.floating_tap_coef
    if param.Floating_RXFFE:
        BEST.floating_tap_locations = THIS.floating_tap_locations
    if OP.RxFFE:
        BEST.RxFFE = THIS.C
        BEST.PSD_results = _value_copy(THIS.PSD_results)
        BEST.MMSE_results = THIS.MMSE_results

    return BEST
