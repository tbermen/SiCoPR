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
from scipy.signal import lfilter


def _TD_CTLE(ir_in, fb, f_z, f_p1, f_p2, kacdc_dB, oversampling):
    # No dtype=float: MATLAB's filter() carries a complex input through, and
    # the cast silently DISCARDED the imaginary part.  atleast_1d because
    # MATLAB filters a scalar (1x1) and returns a scalar, where lfilter raised
    # "selected axis is out of range" on a 0-d array.
    ir_in = np.atleast_1d(np.asarray(ir_in))
    p1_ctle = -2 * np.pi * f_p1
    p2_ctle = -2 * np.pi * f_p2
    z_ctle = -2 * np.pi * f_z * 10 ** (kacdc_dB / 20)
    k_ctle = -p2_ctle
    bilinear_fs = 2 * fb * oversampling
    p2d = (1 + p2_ctle / bilinear_fs) / (1 - p2_ctle / bilinear_fs)
    p1d = (1 + p1_ctle / bilinear_fs) / (1 - p1_ctle / bilinear_fs)
    zd = (1 + z_ctle / bilinear_fs) / (1 - z_ctle / bilinear_fs)
    kd = ((bilinear_fs - z_ctle)
          / ((bilinear_fs - p1_ctle) * (bilinear_fs - p2_ctle))
          * f_p1 / f_z)
    B_filt = k_ctle * kd * np.poly([zd, -1])
    A_filt = np.poly([p1d, p2d])
    # MATLAB filter() runs along the first NON-singleton dimension: down the
    # columns of a matrix, along a row vector.  lfilter defaults to axis=-1,
    # which filtered a matrix along its rows instead.
    axis = 0 if (ir_in.ndim >= 2 and ir_in.shape[0] != 1) else -1
    return lfilter(B_filt, A_filt, ir_in, axis=axis), p1_ctle, p2_ctle, z_ctle


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
        dt = 1.0 / (param.fb * param.samples_per_ui)
        t_end = float(np.asarray(chdata[0].t)[-1])
        new_t = t_end + np.arange(1, n_add + 1) * dt
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
                eq_ir = uneq_ir
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
