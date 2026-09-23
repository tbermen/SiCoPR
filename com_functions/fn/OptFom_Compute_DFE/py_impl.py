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


def _mrange(x, first, step, last, expr):
    """MATLAB `x(first:step:last)` with MATLAB's subscript rules, 1-based.

    An empty range indexes nothing and is legal; every subscript a non-empty
    one produces must be whole, >= 1 and <= numel(x).  A Python slice answers
    all three cases silently -- a negative start reads from the tail and a
    long stop truncates -- so the port returned numbers for calls the
    reference refuses.  COM Octave, OptFom_Compute_DFE:

      ndfe=40 on a 300-sample sbr:
        "error: sbr(361): out of bound 300 (dimensions are 1x300)"
      do_C2M=1, cursor_i=5, T_O=20:
        "error: sbr(-7): subscripts must be either integers 1 to (2^63)-1
         or logicals"
      param.N_tail_start=-2:
        "error: dfetaps(-2): subscripts must be either integers 1 to
         (2^63)-1 or logicals"      (the port took dfetaps[-3:] instead)
    """
    n = int(np.floor((last - first) / step)) + 1 if last >= first else 0
    if n <= 0:
        return x[:0]
    idx = first + step * np.arange(n, dtype=float)
    bad = idx[idx != np.round(idx)]
    if bad.size:
        raise ValueError('OptFom_Compute_DFE: %s(%g): subscripts must be '
                         'positive integers' % (expr, bad[0]))
    if idx[0] < 1:
        raise IndexError('OptFom_Compute_DFE: %s(%d): subscripts must be '
                         'either integers 1 to (2^63)-1 or logicals'
                         % (expr, int(idx[0])))
    if idx[-1] > x.size:
        raise IndexError('OptFom_Compute_DFE: %s(%d): out of bound %d'
                         % (expr, int(idx[-1]), x.size))
    return x[idx.astype(int) - 1]


def OptFom_Compute_DFE(sbr, THIS, param, do_C2M, T_O):
    sbr = np.asarray(sbr, dtype=float)
    cursor_i = int(THIS.cursor_i)  # 0-based
    cursor_1 = cursor_i + 1        # the MATLAB subscript
    # samples_per_ui / ndfe / N_tail_start / T_O go into MATLAB subscript
    # expressions, and MATLAB refuses a fractional subscript.  Truncating them
    # here answered calls the reference declines, so they stay as given and
    # _mrange applies MATLAB's rule.
    M_f = float(param.samples_per_ui)
    ndfe_f = float(param.ndfe)
    N_tail_start = float(param.N_tail_start)

    # Equation 93A-27: DFE cursor samples
    dfecursors = _mrange(sbr, cursor_1 + M_f, M_f,
                         cursor_1 + M_f * ndfe_f, 'sbr')
    M = int(M_f)                   # whole: _mrange above would have refused

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
        dfecursors_windowed = _mrange(sbr, cursor_1 - T_O + M_f, M_f,
                                      cursor_1 + M_f * ndfe_f - T_O, 'sbr')
        excess_dfe_cursors = dfecursors_windowed - actual_dfecursors
    else:
        excess_dfe_cursors = dfecursors - actual_dfecursors

    dfetaps = actual_dfecursors / sbr[cursor_i]

    if len(dfetaps) >= N_tail_start and N_tail_start != 0:
        # MATLAB dfetaps(N_tail_start:end): a negative or fractional
        # N_tail_start is a subscript error, not a slice from the tail.
        tail_taps = _mrange(dfetaps, N_tail_start, 1.0, len(dfetaps),
                            'dfetaps')
        nts0 = int(N_tail_start) - 1    # whole: _mrange would have refused
        tail_RSS = float(np.linalg.norm(tail_taps))
        if tail_RSS != 0:
            if tail_RSS >= param.B_float_RSS_MAX:
                # MATLAB: min(...)*sign(t).*t/tail_RSS -- the division comes
                # LAST.  Factoring it out into a scale first is algebraically
                # the same and not the same in floating point: COM Octave gave
                # use_bmax(3)=0.044776673559449504 where the scale-first form
                # gave 0.04477667355944951, one ulp out.
                min_v = min(tail_RSS, param.B_float_RSS_MAX)
                sgn = np.sign(tail_taps)
                use_bmax = np.asarray(param.use_bmax).ravel().copy()
                use_bmin = np.asarray(param.use_bmin).ravel().copy()
                use_bmax[nts0:] = min_v * sgn * tail_taps / tail_RSS
                use_bmin[nts0:] = min_v * -1 * sgn * tail_taps / tail_RSS
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
