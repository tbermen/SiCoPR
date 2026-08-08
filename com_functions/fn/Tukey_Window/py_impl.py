# ============================================================
# MATLAB→Python translation notes for Tukey_Window
# MATLAB lines: 4677–4696
# ============================================================
# ~exist('fr','var') && ~exist('fb','var'): optional positional args
#   → Python default fr=None, fb=None; both None → use param fields
# Three regions:
#   f < fr  → 1 (passband)
#   fr≤f≤fb → 0.5*cos(2π*(f-fb)/fperiod - π) + 0.5 (raised-cosine rolloff)
#   f > fb  → 0 (stopband)
# H_tw=H_tw(1:length(f)): guard for floating-point edge; Python [:len(f)]
# Boundary check: at f=fr → H=1; at f=fb → H=0 ✓
# ============================================================

import numpy as np


def Tukey_Window(f, param, fr=None, fb=None):
    f = np.asarray(f, dtype=float)
    if fr is None and fb is None:
        fb = param.fb
        fr = param.f_r * param.fb
    fperiod = 2 * (fb - fr)
    H_tw = np.where(
        f < fr,
        1.0,
        np.where(
            (f >= fr) & (f <= fb),
            0.5 * np.cos(2 * np.pi * (f - fb) / fperiod - np.pi) + 0.5,
            0.0,
        ),
    )
    return H_tw[:len(f)]
