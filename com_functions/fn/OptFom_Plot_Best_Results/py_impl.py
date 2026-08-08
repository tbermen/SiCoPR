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
            print(f'CTF peaking gain:  {20*np.log10(float(np.max(np.abs(BEST.ctle_gain)))):.2g} dB')
        except Exception:
            pass
        try:
            Symbol_Adj = int(param.levels) - 1
            print(f'Symbol Available signal:   {float(BEST.cursor)/Symbol_Adj}')
        except Exception:
            pass
