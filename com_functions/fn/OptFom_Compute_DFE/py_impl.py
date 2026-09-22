# ============================================================
# MATLAB→Python translation notes for OptFom_Compute_DFE
# MATLAB lines: 3219–3304
# ============================================================
# cursor_i is 0-based in Python (MATLAB 1-based → Python 0-based).
# dfecursors: MATLAB sbr(cursor_i+M*(1):M:cursor_i+M*(ndfe)) (1-based)
#   → Python sbr[cursor_i+M : cursor_i+M*ndfe+1 : M] (0-based)
# postcursors for floatingDFE: MATLAB sbr(cursor_i+M:M:end) (1-based)
#   → Python sbr[cursor_i+M :: M] (0-based)
# dfecursors_windowed: MATLAB sbr(cursor_i-T_O+M*(1):M:cursor_i+M*ndfe-T_O) (1-based)
#   → Python sbr[cursor_i-T_O+M : cursor_i+M*ndfe-T_O+1 : M] (0-based)
# N_tail_start: 1-based in MATLAB config → subtract 1 for 0-based array access.
# dfe_clipper and floatingDFE inlined as _dfe_clipper and _floatingDFE.
# ============================================================

import numpy as np
from com_functions.fn.dfe_clipper.py_impl import dfe_clipper as _dfe_clipper
from com_functions.fn.findbankloc.py_impl import findbankloc as _findbankloc
from com_functions.fn.floatingDFE.py_impl import floatingDFE as _floatingDFE

def OptFom_Compute_DFE(sbr, THIS, param, do_C2M, T_O):
    sbr = np.asarray(sbr, dtype=float)
    cursor_i = int(THIS.cursor_i)  # 0-based
    M = int(param.samples_per_ui)
    ndfe = int(param.ndfe)
    N_tail_start = int(param.N_tail_start)

    # Equation 93A-27: DFE cursor samples (0-based Python)
    dfecursors = sbr[cursor_i + M : cursor_i + M * ndfe + 1 : M]

    if param.dfe_delta != 0:
        dfecursors_q = (np.floor(np.abs(dfecursors / sbr[cursor_i]) / param.dfe_delta)
                        * param.dfe_delta * np.sign(dfecursors) * sbr[cursor_i])
    else:
        dfecursors_q = dfecursors.copy()

    if param.Floating_DFE:
        postcursors = sbr[cursor_i + M :: M]
        floating_tap_locations, floating_tap_coef, hisi, bmax = _floatingDFE(
            postcursors, param.ndfe_passed, param.N_bf, param.N_bg,
            param.N_bmax, param.bmaxg, sbr[cursor_i], param.dfe_delta)
        bmax_arr = np.asarray(bmax)
        newbmax = np.concatenate([np.asarray(param.bmax).ravel(),
                                   bmax_arr[param.ndfe_passed:param.N_bmax]])
        param.use_bmax = newbmax.reshape(-1, 1) if hasattr(param, 'use_bmax') and np.asarray(param.use_bmax).ndim > 1 else newbmax
        param.use_bmin = np.concatenate([np.asarray(param.bmin).ravel(),
                                          bmax_arr[param.ndfe_passed:param.N_bmax] * -1]).reshape(param.use_bmax.shape) if hasattr(param, 'use_bmax') else np.concatenate([np.asarray(param.bmin).ravel(), bmax_arr[param.ndfe_passed:param.N_bmax] * -1])
    else:
        param.use_bmax = param.bmax
        param.use_bmin = param.bmin
        floating_tap_coef = []

    actual_dfecursors = _dfe_clipper(
        dfecursors_q,
        sbr[cursor_i] * np.asarray(param.use_bmax).ravel(),
        sbr[cursor_i] * np.asarray(param.use_bmin).ravel())

    if do_C2M:
        dfecursors_windowed = sbr[cursor_i - T_O + M : cursor_i + M * ndfe - T_O + 1 : M]
        excess_dfe_cursors = dfecursors_windowed - actual_dfecursors
    else:
        excess_dfe_cursors = dfecursors - actual_dfecursors

    dfetaps = actual_dfecursors / sbr[cursor_i]

    if len(dfetaps) >= N_tail_start and N_tail_start != 0:
        tail_taps = dfetaps[N_tail_start - 1:]  # 1-based → 0-based
        tail_RSS = float(np.linalg.norm(tail_taps))
        if tail_RSS != 0:
            if tail_RSS >= param.B_float_RSS_MAX:
                scale = min(tail_RSS, param.B_float_RSS_MAX) / tail_RSS
                use_bmax = np.asarray(param.use_bmax).ravel().copy()
                use_bmin = np.asarray(param.use_bmin).ravel().copy()
                use_bmax[N_tail_start - 1:] = scale * np.abs(tail_taps)
                use_bmin[N_tail_start - 1:] = -scale * np.abs(tail_taps)
                param.use_bmax = use_bmax
                param.use_bmin = use_bmin

                actual_dfecursors = _dfe_clipper(
                    dfecursors_q,
                    sbr[cursor_i] * np.asarray(param.use_bmax).ravel(),
                    sbr[cursor_i] * np.asarray(param.use_bmin).ravel())
                if do_C2M:
                    excess_dfe_cursors = dfecursors_windowed - actual_dfecursors
                else:
                    excess_dfe_cursors = dfecursors - actual_dfecursors
                dfetaps = actual_dfecursors / sbr[cursor_i]
    else:
        tail_RSS = 0.0

    THIS.dfetaps = dfetaps
    if param.Floating_DFE:
        # floatingDFE returns 0-BASED positions into hisi (see its docstring),
        # but MATLAB stores this field 1-based (ML 3559, where tap_loc indexes
        # hisi 1-based) and both consumers -- the fdfecursors time vector at
        # ML 4133 and the DFE_taps_mV lookup at ML 4143 -- read it that way.
        # The other two producers of this same field, MMSE and force, already
        # return 1-based. Normalise here so the field has ONE base whatever
        # produced it; OptFom_Update_BEST_Post_Optimize converts to subscript.
        THIS.floating_tap_locations = np.asarray(floating_tap_locations) + 1
    THIS.floating_tap_coef = floating_tap_coef
    THIS.tail_RSS = tail_RSS
    THIS.excess_dfe_cursors = excess_dfe_cursors
    return THIS, param
