# ============================================================
# MATLAB→Python translation notes for OptFom_Plot_Best_Results
# MATLAB lines: 3674–3784
# ============================================================
# Plotting function: all figure/plot/stem calls skipped when DISPLAY_WINDOW=False.
# Always prints debug text (display() → print()) regardless of DISPLAY_WINDOW.
# OP.DEBUG != 0: activates display block (irrelevant when DISPLAY_WINDOW=False).
# filter(ones(M,1), 1, uneq_imp_response) → scipy lfilter.
# BEST.cursor_i is 0-based in Python.
# No return value in MATLAB (function signature has no output).
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

from scipy.signal import lfilter


def OptFom_Plot_Best_Results(BEST, t, f, chdata, param, OP):
    """Display/plot best equalizer results (MATLAB lines 3674-3784).

    Plotting skipped (no DISPLAY_WINDOW); debug text always printed when DEBUG!=0.
    No return value.
    """
    M = int(param.samples_per_ui)

    # Always compute PRin for debug reporting
    PRin = lfilter(np.ones(M), [1.0], np.asarray(chdata[0].uneq_imp_response, dtype=float).ravel())
    if len(PRin) < len(np.asarray(BEST.sbr, dtype=float).ravel()):
        PRin = np.concatenate([PRin, np.zeros(len(BEST.sbr) - len(PRin))])

    if OP.DEBUG != 0:
        # Print debug info (replaces MATLAB display() calls)
        try:
            print(f'FOM:                {float(BEST.FOM):.2g} dB')
        except Exception:
            pass
        try:
            print(f'TXFFE coefficients: {np.asarray(BEST.txffe).tolist()}')
        except Exception:
            pass
        try:
            A_p = float(BEST.A_p) if hasattr(BEST, 'A_p') else float(BEST.cursor)
            ISI = float(BEST.ISI) if hasattr(BEST, 'ISI') else 1.0
            print(f'SNR ISI:                {20*np.log10(A_p/ISI):.2g} dB')
        except Exception:
            pass
        try:
            print(f'CTLE DC gain:       {float(BEST.gdc):.4g} dB')
        except Exception:
            pass
        try:
            print(f'CTF peaking gain:  {20*np.log10(float(_mmax(np.abs(BEST.ctle_gain)))):.2g} dB')
        except Exception:
            pass
        try:
            Symbol_Adj = int(param.levels) - 1
            print(f'Symbol Available signal:   {float(BEST.cursor)/Symbol_Adj}')
        except Exception:
            pass
