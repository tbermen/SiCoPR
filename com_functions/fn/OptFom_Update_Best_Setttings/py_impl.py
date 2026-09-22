import numpy as np
from com_functions.fn.FFE.py_impl import FFE as _FFE
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


def OptFom_Update_Best_Setttings(BEST, THIS, sbr, chdata, param, OP):
    """Copy THIS/param fields into BEST after a successful FOM update (MATLAB lines 3935-3975).

    Note: function name preserves original MATLAB typo (three t's).
    Returns BEST.
    """
    BEST.ffegain = param.current_ffegain
    BEST.txffe = _value_copy(THIS.txffe)
    BEST.txffe_index = _value_copy(THIS.tx_index_vector)
    BEST.sbr = _value_copy(sbr)
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
    BEST.H_ctf = _value_copy(getattr(THIS, 'H_ctf', None))
    BEST.sdd21ctf = _value_copy(getattr(chdata[0], 'sdd21ctf', None))

    BEST.sigma_N = _value_copy(THIS.sigma_N)
    BEST.h_J = _value_copy(THIS.h_J)
    BEST.A_s = THIS.A_s
    BEST.A_p = THIS.A_p
    BEST.ISI = _value_copy(THIS.ISI_N)
    BEST.bmax = _value_copy(param.use_bmax)
    BEST.bmin = _value_copy(param.use_bmin)
    BEST.tail_RSS = THIS.tail_RSS
    BEST.dfetaps = _value_copy(THIS.dfetaps)

    if param.Floating_DFE:
        BEST.floating_tap_locations = _value_copy(THIS.floating_tap_locations)
        BEST.floating_tap_coef = _value_copy(THIS.floating_tap_coef)
    if param.Floating_RXFFE:
        BEST.floating_tap_locations = _value_copy(THIS.floating_tap_locations)
    if OP.RxFFE:
        BEST.RxFFE = _value_copy(THIS.C)
        BEST.PSD_results = _value_copy(THIS.PSD_results)
        BEST.MMSE_results = _value_copy(THIS.MMSE_results)

    return BEST
