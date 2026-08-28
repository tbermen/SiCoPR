# ============================================================
# MATLAB→Python translation notes for OptFom_Compute_RxFFE
# MATLAB lines: 3305–3369
# ============================================================
# MMSE path: get_PSDs -> MMSE solver. Other (force) path: force() solver.
#   Both deps are implemented; get_PSDs is called with the real S_RN/S_IN/H_interp
#   (its injected-fn defaults are stubs in the assembled module).
# RXFFE_Illegal, FFE: implemented (top-level in the assembled module).
# OP.FFE_OPT_METHOD: 'MMSE' -> MMSE path; otherwise force path.
# cursor_i is 0-based Python (same convention as rest of OptFom_ family).
# THIS fields set: C, floating_tap_locations, FOM, PSD_results, MMSE_results.
# skip_it: 1 if RXFFE_Illegal and not RxFFE_with_MMSE.
# Callees (get_PSDs, MMSE, force, RXFFE_Illegal, FFE, S_RN, S_IN, H_interp) are
# top-level functions in the assembled sicopr.py — integration-tested, not standalone.
# ============================================================

import numpy as np
from types import SimpleNamespace


def OptFom_Compute_RxFFE(sbr, THIS, Noise_XC, chdata, param, OP):
    skip_it = 0
    PSD_results = THIS.PSD_results
    txffe = THIS.txffe
    cursor_i = THIS.cursor_i
    g_dc = THIS.g_dc
    g_DC_low = THIS.g_DC_low

    if str(OP.FFE_OPT_METHOD).upper() == 'MMSE':
        OP.WO_TXFFE = 0
        PSD_results = get_PSDs(PSD_results, sbr, cursor_i, txffe, g_dc, g_DC_low,
                               param, chdata, OP,
                               _S_RN_fn=S_RN, _S_IN_fn=S_IN, _H_interp_fn=H_interp)
        MMSE_results = MMSE(PSD_results, sbr, cursor_i, param, OP)
        C = MMSE_results.C
        floating_tap_locations = MMSE_results.floating_tap_locations
        FOM = MMSE_results.FOM
    else:
        # force(): MATLAB [~, C, floating_tap_locations] = force(sbr,param,OP,cursor_i,[],0,chdata,txffe,Noise_XC)
        _, C, floating_tap_locations = force(sbr, param, OP, cursor_i, None, 0,
                                             chdata, txffe, Noise_XC)
        MMSE_results = None
        FOM = 0

    # RXFFE legality check (force() also checks when backoff is enabled).
    if not getattr(OP, 'RxFFE_with_MMSE', 0):
        if RXFFE_Illegal(C, param):
            skip_it = 1
            return sbr, THIS, skip_it

    # Apply RxFFE to the pulse only after the legality check (FFE is expensive).
    sbr = FFE(C, int(param.RxFFE_cmx), int(param.samples_per_ui), sbr)

    THIS.C = C
    THIS.floating_tap_locations = floating_tap_locations
    THIS.FOM = FOM
    THIS.PSD_results = PSD_results
    THIS.MMSE_results = MMSE_results
    return sbr, THIS, skip_it
