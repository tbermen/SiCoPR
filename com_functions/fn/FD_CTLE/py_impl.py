# ============================================================
# MATLAB→Python translation notes for FD_CTLE
# MATLAB lines: 1681–1683
# ============================================================
# 10.^(kacdc_dB/20): scalar → float; 10**(kacdc_dB/20)
# ./ and .* are element-wise → standard numpy ops on arrays
# kacdc_dB ≤ 0 typical: DC gain < 1, peaking at high frequency
# At freq=0: H = 10^(kacdc_dB/20) (real DC value)
# ============================================================

import numpy as np


def FD_CTLE(freq, f_z, f_p1, f_p2, kacdc_dB):
    freq = np.asarray(freq, dtype=float)
    num = 10 ** (kacdc_dB / 20) + 1j * freq / f_z
    den = (1 + 1j * freq / f_p1) * (1 + 1j * freq / f_p2)
    return num / den
