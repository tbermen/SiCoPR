import numpy as np
from types import SimpleNamespace

def _value_copy(obj):
    """MATLAB assigns structs and arrays BY VALUE; Python binds a reference.

    A snapshot of the winning candidate must therefore DETACH from the live
    candidate, or later evaluation silently rewrites what was recorded. That is
    defect #9: get_PSDs mutates its `result` argument in place and optimize_fom
    reuses one PSD_results object per CTLE block, so BEST ended up holding the
    LAST tick swept rather than the winning one, and the COM stage copies those
    arrays straight into the reported noise (ML 547-553).

    Recursive and type-general so it can be applied to every field uniformly:
    scalars and strings are immutable and pass through untouched, so wrapping a
    field costs nothing and removes the need to reason field-by-field about
    whether something downstream mutates it.

    tests/test_snapshot_isolation.py holds the invariant this exists to satisfy.
    """
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, list):
        return [_value_copy(e) for e in obj]
    if isinstance(obj, tuple):
        return tuple(_value_copy(e) for e in obj)
    if isinstance(obj, dict):
        return {k: _value_copy(v) for k, v in obj.items()}
    if isinstance(obj, SimpleNamespace):
        out = SimpleNamespace()
        for k, v in vars(obj).items():
            setattr(out, k, _value_copy(v))
        return out
    return obj


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
    BEST.bmax = _value_copy(param.bmax)
    BEST.bmin = _value_copy(param.bmin)
    BEST.tail_RSS = 0.0
    BEST.ffegain = 0.0
    BEST.txffe = _value_copy(THIS.txffe)
    BEST.sbr = _value_copy(sbr)
    BEST.ctle = THIS.ctle_index

    if OP.RxFFE:
        BEST.PSD_results = _value_copy(THIS.PSD_results)
        BEST.MMSE_results = _value_copy(THIS.MMSE_results)
        BEST.RxFFE = _value_copy(THIS.C)

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

    BEST.sigma_N = _value_copy(THIS.sigma_N)
    BEST.h_J = _value_copy(THIS.h_J)
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
