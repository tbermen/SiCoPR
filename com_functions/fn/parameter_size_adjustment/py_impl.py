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


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)



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
        PORTZ_mult = np.ones(int(_mmax(pkg_sel)), dtype=float)

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
