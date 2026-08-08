# ============================================================
# MATLAB→Python translation notes for OptFom_Calc_Hr
# MATLAB lines: 2874–2880
# ============================================================
# Calls Bessel_Thomson_Filter, Butterworth_Filter, Raised_Cosine_Filter —
#   all sibling py_impl.py files; protocol forbids cross-module imports →
#   full implementations inlined as private helpers _bt, _bw, _rc.
# OP struct fields: OP.Bessel_Thomson, OP.Butterworth, OP.Raised_Cosine (bool)
# H_r = product of three filter responses (element-wise multiply)
# param fields needed: fb, fb_BW_cutoff (BW), BTorder/fb_BT_cutoff (BT),
#   RC_Start/RC_end (RC)
# ============================================================

import math
import numpy as np

_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]


def _bessel_poly(n):
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        a[ii] = (math.factorial(2 * n - ii)
                 / (2 ** (n - ii) * math.factorial(ii) * math.factorial(n - ii)))
    return a


def _butterworth(param, f, use_BW):
    if not use_BW:
        return np.ones(len(f))
    s = 1j * f / (param.fb_BW_cutoff * param.fb)
    return 1.0 / np.polyval(_BW_POLY, s)


def _bessel_thomson(param, f, use_BT):
    if not use_BT:
        return np.ones(len(f))
    a = _bessel_poly(param.BTorder)
    acoef = a[::-1]
    s = 1j * f / (param.fb_BT_cutoff * param.fb)
    return a[0] / np.polyval(acoef, s)


def _tukey_window(f, fr, fb_top):
    fperiod = 2 * (fb_top - fr)
    H = np.where(
        f < fr, 1.0,
        np.where(
            (f >= fr) & (f <= fb_top),
            0.5 * np.cos(2 * np.pi * (f - fb_top) / fperiod - np.pi) + 0.5,
            0.0,
        ),
    )
    return H[:len(f)]


def _raised_cosine(param, f, use_RC):
    if not use_RC:
        return np.ones(len(f))
    return _tukey_window(f, param.RC_Start, param.RC_end)


def OptFom_Calc_Hr(f, param, OP):
    f = np.asarray(f, dtype=float)
    H_bt = _bessel_thomson(param, f, OP.Bessel_Thomson)
    H_bw = _butterworth(param, f, OP.Butterworth)
    H_RCos = _raised_cosine(param, f, OP.Raised_Cosine)
    return H_bw * H_bt * H_RCos


if __name__ == "__main__":
    from types import SimpleNamespace
    param = SimpleNamespace(fb=25e9, fb_BW_cutoff=1.0,
                            BTorder=2, fb_BT_cutoff=1.0,
                            RC_Start=5e9, RC_end=20e9)
    OP = SimpleNamespace(Bessel_Thomson=False, Butterworth=False, Raised_Cosine=False)
    H = OptFom_Calc_Hr(np.array([0.0, 10e9, 25e9]), param, OP)
    print("All disabled:", H)
