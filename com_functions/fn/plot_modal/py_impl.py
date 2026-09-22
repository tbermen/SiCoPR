# ============================================================
# MATLAB→Python translation notes for plot_modal
# MATLAB lines: 8914–9010
# ============================================================
# Computes CM_MASK_REPORT struct (modal mask pass/fail) when OP.CM_MASK_REPORT.
# All MATLAB figure/plot calls skipped (OP.DISPLAY_WINDOW=False).
# Sfield: '_orig' if not OP.SHOW_BRD, else '_raw'.
# RLcc_mask, RLcc178, RLdc_mask: lambda functions of f/1e9.
# chdata[0].faxis is frequency axis (Hz).
# dB(x) = 20*log10(abs(x)).
# Return value: return_struct (SimpleNamespace) or None if not CM_MASK_REPORT.
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

from types import SimpleNamespace


def _RLcc_mask(f_ghz):
    f = np.asarray(f_ghz, dtype=float)
    return (
        -2.0 * ((5 >= f) & (f < 4))
        - (2 + 2 / 40 * (f - 4)) * ((4 <= f) & (f < 44))
        - (4 + 2 / 16 * (44 - f)) * ((44 <= f) & (f < 60))
        - 2.0 * ((60 <= f) & (f <= 67))
        - 2.0 * (f > 67)
    )


def _RLcc178(f_ghz):
    f = np.asarray(f_ghz, dtype=float)
    return -np.ones(len(f)) * 3.25


def _RLdc_mask(f_ghz):
    f = np.asarray(f_ghz, dtype=float)
    return (
        (-23 + 22 * f / 106.25) * ((f >= 0.05) & (f < 53.125))
        - 12.0 * ((f >= 53.125) & (f <= 67))
        - 12.0 * (f > 67)
    )


def _dB(x):
    return 20.0 * np.log10(np.squeeze(np.abs(np.asarray(x, dtype=complex))) + np.finfo(float).eps)


def plot_modal(param, OP, chdata):
    """Modal mask computation and (skipped) plotting (MATLAB lines 8914-9010).

    Returns SimpleNamespace with Rlcc/Rldc margin fields when OP.CM_MASK_REPORT,
    otherwise returns None.
    """
    if isinstance(chdata, list):
        ch0 = chdata[0]
    else:
        ch0 = chdata

    Sfield = '_raw' if getattr(OP, 'SHOW_BRD', False) else '_orig'
    f = np.asarray(ch0.faxis, dtype=float)
    f_ghz = f / 1e9

    if getattr(OP, 'CM_MASK_REPORT', False):
        rs = SimpleNamespace()

        scc11 = np.asarray(getattr(ch0, 'scc11' + Sfield, np.zeros(len(f))), dtype=complex)
        scd22 = np.asarray(getattr(ch0, 'scd22' + Sfield, np.zeros(len(f))), dtype=complex)
        sdc22 = np.asarray(getattr(ch0, 'sdc22' + Sfield, np.zeros(len(f))), dtype=complex)

        rs.Rlcc_179mm = _RLcc_mask(f_ghz) - _dB(scc11)
        rs.Rlcc_179mm_fail = bool(_mmin(rs.Rlcc_179mm) < 0)
        rs.Rlcc_178mm = _RLcc178(f_ghz) - _dB(scc11)
        rs.Rlcc_178mm_fail = bool(_mmin(rs.Rlcc_178mm) < 0)
        rs.Rlcd_179mm = _RLdc_mask(f_ghz) - _dB(scd22)
        rs.Rlcd_179mm_fail = bool(_mmin(rs.Rlcd_179mm) < 0)
        rs.Rldc_179mm = _RLdc_mask(f_ghz) - _dB(sdc22)
        rs.Rldc_179mm_fail = bool(_mmin(rs.Rldc_179mm) < 0)
        return rs

    return None
