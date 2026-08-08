# ============================================================
# MATLAB→Python translation notes for parameter_size_adjustment
# MATLAB lines: 8794–8845
# ============================================================
# Expands scalar parameters to arrays of the required length.
# z_p_tx_cases is a 2-D array; ncases = number of rows.
# pkg_sel_vec length = max(OP.pkg_len_select).
# Uses np.ones_like to match the shape/dtype of reference arrays.
# ============================================================

import numpy as np


def parameter_size_adjustment(param, OP):
    make_length2 = ['C_pkg_board', 'C_diepad', 'L_comp', 'C_bump', 'tfx',
                    'C_v', 'C_0', 'C_1', 'pkg_Z_c', 'brd_Z_c', 'R_diepad']
    make_length_WCPORTZ = ['a_thru', 'a_fext', 'a_next', 'SNDR']
    make_length_GDC = ['CTLE_fp1', 'CTLE_fp2', 'CTLE_fz', 'f_HP_Z', 'f_HP_P']
    make_length_DCHP = ['f_HP']
    make_length_ncases = ['AC_CM_RMS']

    ncases = np.asarray(param.z_p_tx_cases).shape[0]

    pkg_sel = np.asarray(OP.pkg_len_select)
    if OP.WC_PORTZ:
        PORTZ_mult = np.ones(2, dtype=float)
    else:
        PORTZ_mult = np.ones(int(np.max(pkg_sel)), dtype=float)

    for field in make_length2:
        raw = getattr(param, field, None)
        if raw is None:
            continue
        val = np.asarray(raw)
        if val.size == 1:
            setattr(param, field, float(val.ravel()[0]) * np.ones(2))

    for field in make_length_ncases:
        raw = getattr(param, field, None)
        if raw is None:
            continue
        val = np.asarray(raw)
        if val.size == 1:
            setattr(param, field, float(val.ravel()[0]) * np.ones(ncases))

    for field in make_length_GDC:
        raw = getattr(param, field, None)
        if raw is None:
            continue
        val = np.asarray(raw)
        if val.size == 1:
            ref = np.asarray(param.ctle_gdc_values)
            setattr(param, field, float(val.ravel()[0]) * np.ones_like(ref, dtype=float))

    for field in make_length_DCHP:
        raw = getattr(param, field, None)
        if raw is None:
            continue
        val = np.asarray(raw)
        if val.size == 1:
            ref = np.asarray(param.g_DC_HP_values)
            setattr(param, field, float(val.ravel()[0]) * np.ones_like(ref, dtype=float))

    for field in make_length_WCPORTZ:
        raw = getattr(param, field, None)
        if raw is None:
            continue
        val = np.asarray(raw)
        if val.size == 1:
            setattr(param, field, float(val.ravel()[0]) * PORTZ_mult)

    return param
