import numpy as np
from com_functions.fn.FFE.py_impl import FFE as _FFE

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


def _margmax(a):
    """MATLAB's `[~,i] = max(x)`: a NaN is SKIPPED unless every element is NaN,
    in which case the index is 1.  np.argmax returns the index of the first NaN
    instead, so one bad sample moved the cursor to the head of the response.

    COM Octave: max([NaN 1 3 2]) -> 3 at 1-based index 3   (np.argmax gave 0)
                max([1 NaN 5])   -> 5 at 1-based index 3   (np.argmax gave 1)
                max([NaN NaN])   -> NaN at 1-based index 1 (np.argmax gave 0)
    """
    a = np.asarray(a)
    if a.dtype.kind != 'f':
        return int(np.argmax(a))
    nan = np.isnan(a)
    if nan.all():
        return 0
    if nan.any():
        return int(np.nanargmax(a))
    return int(np.argmax(a))


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
        cursor_i = _margmax(sbr)
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
    BEST.A_p = float(_mmax(sbr))
    BEST.ISI = 1.0

    # DFE taps: UI-spaced postcursors 1..ndfe, normalized by cursor
    M = int(param.samples_per_ui)
    ndfe = int(param.ndfe)
    ci = int(cursor_i)
    cursor_val = float(sbr[ci])
    # MATLAB: sbr(cursor_i+M:M:cursor_i+M*ndfe)/sbr(cursor_i) — both 1-based
    # Python: sbr[ci+M : ci+M*ndfe+1 : M] / cursor_val (0-based ci)
    # A python slice silently stops at the end of the array; MATLAB refuses, so
    # a short response used to yield FEWER dfe taps than param.ndfe without a
    # word.  COM Octave: sbr 1x20, cursor_i=5, M=4, ndfe=6 ->
    #   "error: sbr(29): out of bound 20 (dimensions are 1x20)".
    if ci + M * ndfe >= len(sbr):
        raise IndexError(
            'OptFom_Update_Best_Settings_EQ_Failed: sbr(%d): out of bound %d - '
            'the pulse response is too short for %d DFE taps'
            % (ci + M * ndfe + 1, len(sbr), ndfe))
    BEST.dfetaps = sbr[ci + M: ci + M * ndfe + 1: M] / cursor_val
    BEST.A_s = cursor_val

    if param.Floating_DFE:
        BEST.floating_tap_locations = []
        BEST.floating_tap_coef = []

    return BEST
