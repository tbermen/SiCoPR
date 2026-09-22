# ============================================================
# MATLAB→Python translation notes for Apply_EQ
# MATLAB lines: 905–976
# ============================================================
# Inlines TD_CTLE (4593–4609) and FFE (2026–2048).
# MATLAB 1-based index fom_result.ctle → Python [ctle-1].
# sample_start: MATLAB mod(t_s-1, M)+1 with t_s 1-BASED. Python's fom_result.t_s
#   is BEST.cursor_i, already 0-based, so t_s_m = t_s_py + 1 and the 0-based
#   start is mod(t_s_m - 1, M) = t_s_py % M. Same identity get_PSDs uses
#   (cursor_i % M for MATLAB's mod(cursor_i-1,M)+1). An earlier version wrote
#   (t_s - 1) % M, converting the RESULT to 0-based but not the INPUT, and so
#   sampled the ADC-clip pulse one sample early.
# OP.RxFFE branch calls the real force() to apply the Rx-FFE taps to the pulse.
# chdata extended with zeros if shorter than fom_result.sbr.
# ============================================================

import numpy as np
from com_functions.fn.FFE.py_impl import FFE as _FFE
from com_functions.fn.TD_CTLE.py_impl import TD_CTLE as _TD_CTLE
from scipy.signal import lfilter


def Apply_EQ(param, fom_result, chdata, OP):
    FB = param.fb
    ctle_idx = int(fom_result.ctle) - 1  # MATLAB 1-based → 0-based
    FZ = np.asarray(param.CTLE_fz, dtype=float)[ctle_idx]
    FP1 = np.asarray(param.CTLE_fp1, dtype=float)[ctle_idx]
    FP2 = np.asarray(param.CTLE_fp2, dtype=float)[ctle_idx]
    GDC = np.asarray(param.ctle_gdc_values, dtype=float)[ctle_idx]

    hp_idx = int(fom_result.best_G_high_pass) - 1  # MATLAB 1-based → 0-based
    f_HP = getattr(param, 'f_HP', None)
    g_DC_HP_values = getattr(param, 'g_DC_HP_values', None)
    f_HP_Z = getattr(param, 'f_HP_Z', None)
    f_HP_P = getattr(param, 'f_HP_P', None)

    FHP = float(np.asarray(f_HP, dtype=float)[hp_idx]) if (f_HP is not None and len(f_HP) > 0) else None
    GDCHP = float(np.asarray(g_DC_HP_values, dtype=float)[hp_idx]) if (g_DC_HP_values is not None and len(g_DC_HP_values) > 0) else None
    FHPZ = float(np.asarray(f_HP_Z, dtype=float)[ctle_idx]) if (f_HP_Z is not None and len(f_HP_Z) > 0) else None
    FHPP = float(np.asarray(f_HP_P, dtype=float)[ctle_idx]) if (f_HP_P is not None and len(f_HP_P) > 0) else None

    # Extend chdata[0] if needed
    SBR_Len = len(fom_result.sbr)
    if len(np.asarray(chdata[0].uneq_imp_response)) < SBR_Len:
        n_add = SBR_Len - len(chdata[0].uneq_imp_response)
        t_end = float(np.asarray(chdata[0].t)[-1])
        # MATLAB L927: (1:samples_added)/param.fb/param.samples_per_ui + t(end)
        # -- TWO successive divisions, not one multiply by a pre-formed dt.
        # (k/fb)/M and k*(1/(fb*M)) are not the same double: 2 of 7 and 3 of 13
        # sample offsets differ at fb=25e9, M=8.  COM Octave, 7 added samples,
        # t(end)=1.95e-10: last t is 2.3000000000000001e-10, where the
        # pre-formed-dt version gave 2.2999999999999998e-10.
        new_t = np.arange(1, n_add + 1) / param.fb / param.samples_per_ui + t_end
        chdata[0].uneq_imp_response = np.concatenate([np.asarray(chdata[0].uneq_imp_response), np.zeros(n_add)])
        chdata[0].uneq_pulse_response = np.concatenate([np.asarray(chdata[0].uneq_pulse_response), np.zeros(n_add)])
        chdata[0].t = np.concatenate([np.asarray(chdata[0].t), new_t])

    M = param.samples_per_ui
    for i in range(param.number_of_s4p_files):
        uneq_ir = np.asarray(chdata[i].uneq_imp_response, dtype=float)
        if OP.INCLUDE_CTLE == 1:
            ctle_type = param.CTLE_type
            if ctle_type == 'CL93':
                eq_ir, _, _, _ = _TD_CTLE(uneq_ir, FB, FZ, FP1, FP2, GDC, M)
            elif ctle_type == 'CL120d':
                eq_ir, _, _, _ = _TD_CTLE(uneq_ir, FB, FZ, FP1, FP2, GDC, M)
                eq_ir, _, _, _ = _TD_CTLE(eq_ir, FB, FHP, FHP, 100e100, GDCHP, M)
            elif ctle_type == 'CL120e':
                eq_ir, _, _, _ = _TD_CTLE(uneq_ir, FB, FZ, FP1, FP2, GDC, M)
                eq_ir, _, _, _ = _TD_CTLE(eq_ir, FB, FHPZ, FHPP, 1e99, 0, M)
            else:
                # MATLAB's switch (L933-942) has no `otherwise`, so an
                # unrecognised CTLE_type leaves eq_ir undefined and the next
                # line errors.  Falling through to the uneq response answered
                # an un-equalised channel as though it were equalised.
                # COM Octave, param.CTLE_type='NOSUCH', OP.INCLUDE_CTLE=1:
                #   "error: 'eq_ir' undefined near line 46, column 31"
                raise ValueError("Apply_EQ: unsupported param.CTLE_type %r "
                                 "(MATLAB leaves eq_ir undefined)" % (ctle_type,))
        else:
            eq_ir = uneq_ir

        chdata[i].eq_imp_response = eq_ir
        eq_pulse = lfilter(np.ones(M), [1.0], eq_ir)

        if chdata[i].type in ('FEXT', 'THRU'):
            eq_pulse = _FFE(fom_result.txffe, fom_result.cur - 1, M, eq_pulse)

        chdata[i].pulse_response_w_CFT_TXFFE_noRxFFE = eq_pulse
        chdata[i].ctle_pulse = eq_pulse
        sample_start = int(fom_result.t_s) % M  # 0-based; see header note
        chdata[i].pulse_sampled_w_tx_ffe_ctle = eq_pulse[sample_start::M]
        chdata[i].t_sampled_w_tx_ffe_ctle = np.asarray(chdata[i].t)[sample_start::M]

        if OP.RxFFE:
            if OP.FFE_OPT_METHOD.upper() == 'MMSE':
                chdata[i].ctle_imp_response = _FFE(fom_result.RxFFE, fom_result.cur - 1, M, eq_ir)
            # MATLAB L973: [eq_pulse, C] = force(eq_pulse, param, OP, t_s, fom_result.RxFFE)
            # C provided -> force applies the precomputed RxFFE taps (top-level fn).
            eq_pulse, _, _ = force(eq_pulse, param, OP, fom_result.t_s, fom_result.RxFFE)
        chdata[i].eq_pulse_response = eq_pulse

    return chdata
