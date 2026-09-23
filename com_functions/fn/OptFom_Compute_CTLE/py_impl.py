# ============================================================
# MATLAB→Python translation notes for OptFom_Compute_CTLE
# MATLAB lines: 3144–3218
# ============================================================
# FD_CTLE and TD_CTLE inlined as _FD_CTLE and _TD_CTLE.
# OptFom_FD_or_TD_Fields inlined: TDMODE→'uneq_pulse_response'/'ctle_pulse_response'.
# THIS.ctle_index and THIS.g_LP_index are 1-based (MATLAB convention) →
#   subtract 1 for 0-based array access.
# ir_last: MATLAB find(...,1,'last') returns 1-based; Python np.where returns 0-based,
#   exclusive end = ir_last + 1.
# RX_CALIBRATION: uses chdata[1] (index 1 = MATLAB chdata(2)).
# ============================================================

import numpy as np
from com_functions.fn.FD_CTLE.py_impl import FD_CTLE as _FD_CTLE
from com_functions.fn.TD_CTLE.py_impl import TD_CTLE as _TD_CTLE

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


from scipy.signal import lfilter
from types import SimpleNamespace


def OptFom_Compute_CTLE(chdata, ctle_gain, THIS, f_xc, param, OP):
    f = np.asarray(chdata[0].faxis, dtype=float)
    ctle_index = int(THIS.ctle_index) - 1  # 0-based
    g_LP_index = int(THIS.g_LP_index) - 1  # 0-based
    baud_rate = 1.0 / param.ui
    gdc_values = np.asarray(param.ctle_gdc_values)
    g_dc = float(gdc_values.ravel()[ctle_index])
    CTLE_fp1 = float(np.asarray(param.CTLE_fp1).ravel()[ctle_index])
    CTLE_fp2 = float(np.asarray(param.CTLE_fp2).ravel()[ctle_index])
    CTLE_fz = float(np.asarray(param.CTLE_fz).ravel()[ctle_index])
    g_DC_HP_values = np.asarray(param.g_DC_HP_values)
    f_xc_arr = np.asarray(f_xc, dtype=float)

    uneq_field = 'uneq_pulse_response' if OP.TDMODE else 'uneq_imp_response'
    ctle_field = 'ctle_pulse_response' if OP.TDMODE else 'ctle_imp_response'

    ctle_type = param.CTLE_type
    g_DC_low = None
    f_HP = None
    HP_Z = None
    HP_P = None

    if ctle_type == 'CL93':
        # MATLAB sets both to the SCALAR 1, and H_low_xc is a return value.
        # COM Octave, CTLE_type='CL93' with a 3-point f_xc: size(H_low_xc) is
        # 1x1, not 1x3.  H_ctf is unchanged either way (1.0*ctle_gain is
        # exact), but the port handed the caller a vector the length of f_xc.
        H_low = 1.0
        H_low_xc = 1.0
    elif ctle_type == 'CL120d':
        g_DC_low = float(g_DC_HP_values.ravel()[g_LP_index])
        f_HP = float(np.asarray(param.f_HP).ravel()[g_LP_index])
        H_low = _FD_CTLE(f, f_HP, f_HP, 100e100, g_DC_low)
        H_low_xc = _FD_CTLE(f_xc_arr, f_HP, f_HP, 100e100, g_DC_low)
    elif ctle_type == 'CL120e':
        HP_Z = float(np.asarray(param.f_HP_Z).ravel()[ctle_index])
        HP_P = float(np.asarray(param.f_HP_P).ravel()[ctle_index])
        H_low = _FD_CTLE(f, HP_Z, HP_P, 100e100, 0.0)
        H_low_xc = _FD_CTLE(f_xc_arr, HP_Z, HP_P, 100e100, 0.0)
    else:
        raise ValueError(f'Unknown CTLE_type: {ctle_type}')

    H_ctf = H_low * np.asarray(ctle_gain)

    if OP.INCLUDE_CTLE == 1:
        for k in range(param.num_s4p_files):
            ir = np.asarray(getattr(chdata[k], uneq_field), dtype=float)
            # MATLAB max([]) is [], and `[] > []*thr` is empty, so an empty
            # response reaches the find() below without erroring.  COM Octave,
            # chdata(1).uneq_pulse_response = []: both fields come back 1x0.
            # np.max raised "zero-size array to reduction operation maximum".
            ir_peak = float(_mmax(np.abs(ir))) if ir.size else 0.0
            last_arr = np.where(np.abs(ir) > ir_peak * OP.impulse_response_truncation_threshold)[0]
            # find(...,1,'last') is EMPTY when nothing clears the threshold --
            # an all-zero response, or a truncation threshold of 1 or more --
            # and MATLAB then evaluates ir(1:[]), which is EMPTY, not the whole
            # vector.  COM Octave, uneq_pulse_response all zeros: uneq and ctle
            # both come back 1x0.  Leaving ir alone kept all 64 samples and ran
            # the CTLE over them.
            ir = ir[:int(last_arr[-1]) + 1] if len(last_arr) > 0 else ir[:0]
            setattr(chdata[k], uneq_field, ir)
            ctle_out, _, _, _ = _TD_CTLE(ir, baud_rate, CTLE_fz, CTLE_fp1, CTLE_fp2,
                                          g_dc, param.samples_per_ui)
            if ctle_type == 'CL120d':
                ctle_out, _, _, _ = _TD_CTLE(ctle_out, baud_rate, f_HP, f_HP, 100e100,
                                              g_DC_low, param.samples_per_ui)
            elif ctle_type == 'CL120e':
                ctle_out, _, _, _ = _TD_CTLE(ctle_out, baud_rate, HP_Z, HP_P, 100e100,
                                              0.0, param.samples_per_ui)
            setattr(chdata[k], ctle_field, ctle_out)
    else:
        for k in range(param.num_s4p_files):
            setattr(chdata[k], ctle_field, getattr(chdata[k], uneq_field))

    for k in range(param.num_s4p_files):
        chdata[k].sdd21ctf = np.asarray(chdata[k].sdd21) * H_ctf

    if OP.RX_CALIBRATION:
        f2 = np.asarray(chdata[1].faxis, dtype=float)
        ctle_gain2 = _FD_CTLE(f2, CTLE_fz, CTLE_fp1, CTLE_fp2, g_dc)
        if ctle_type == 'CL93':
            H_low2 = 1.0                       # MATLAB `H_low2=1`, a scalar
        elif ctle_type == 'CL120d':
            H_low2 = _FD_CTLE(f2, f_HP, f_HP, 100e100, g_DC_low)
        else:  # CL120e
            H_low2 = _FD_CTLE(f2, HP_Z, HP_P, 100e100, 0.0)
        H_ctf2 = H_low2 * ctle_gain2
    else:
        H_ctf2 = 1

    return chdata, H_ctf, H_low_xc, H_ctf2
