"""Audit batch B15 spot-check: MLSE Gaussian SNR_DFE_eqivalent (MATLAB 2271-2352
-> com.py 2713-2783).

B15-D19 (low, diagnostic-only): in the Gaussian equivalent-SNR step, MATLAB
line 2311 has
    qfunc( (1-2*alpha)*main/(L-1)*sigma_noise )
which, by MATLAB's left-associative */ precedence, puts sigma_noise in the
NUMERATOR: (1-2*alpha)*main*sigma_noise/(L-1). com.py writes
    _MLSE__qfunc((1-2*alpha)*A_peak/((L-1)*sigma_noise))
with sigma_noise in the DENOMINATOR. (The commented-out reference at MATLAB 2295
divides, so MATLAB 2311 looks like a typo - but the audit reproduces MATLAB.)

Impact: only the reported diagnostics SNR_DFE_eqivalent_Gaussian and
delta_com_Gaussian differ. The operative COM is unaffected: MATLAB sets BOTH
COM_Gaussian and COM_CDF to new_com_CDF (the CDF path, 2345-2346), and com.py
does the same (2778-2779). So no COM value changes.

Run: python tests/test_mlse.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import com  # noqa: E402


# --- 1. Root-cause: the two qfunc-argument forms differ (precedence) ----------
alpha, main, L, sigma = 0.3, 0.1, 4, 0.02
ml_arg = (1 - 2 * alpha) * main / (L - 1) * sigma          # MATLAB 2311 (left-assoc)
py_arg = (1 - 2 * alpha) * main / ((L - 1) * sigma)         # com.py form
# EXPECTED FAIL: com.py form != MATLAB form -> documents B15-D19.
xcheck("mlse_gaussian_qfunc_arg_matches_matlab",
      abs(ml_arg - py_arg) <= 1e-15,
      "DIVERGENT (B15-D19, low, diagnostic-only): MATLAB arg=%.6e "
      "((1-2a)*main*sigma/(L-1)) vs com.py arg=%.6e ((1-2a)*main/((L-1)*sigma)); "
      "ratio=%.4e == sigma^2=%.4e. com.py 2749/MATLAB 2311." %
      (ml_arg, py_arg, py_arg / ml_arg, sigma ** 2))
check("mlse_gaussian_arg_ratio_is_sigma_squared",
      abs((ml_arg / py_arg) - sigma ** 2) <= 1e-12,
      "the MATLAB/com.py arg ratio is not sigma^2 (mechanism check)")


# --- 2. MLSE runs and the operative COM is diagnostic-independent -------------
# Build a symmetric noise PDF via the audited normal_dist, and a CDF (cumsum).
pdf = com.normal_dist(0.02, 7, 1e-4)          # sigma=0.02
cdf = np.cumsum(np.asarray(pdf.y, dtype=float))
param = SimpleNamespace(levels=4, specBER=1e-5)
res = com.MLSE(param, 0.3, 0.10, 0.05, pdf, cdf)   # A_s >= A_ni -> MLSE applied

check("mlse_COM_Gaussian_equals_COM_CDF",
      abs(res.COM_Gaussian - res.COM_CDF) <= 1e-12,
      "COM_Gaussian != COM_CDF; both should be new_com_CDF (MATLAB 2345-2346), so "
      "the B15-D19 Gaussian-arg divergence must not reach COM: %.6f vs %.6f"
      % (res.COM_Gaussian, res.COM_CDF))
check("mlse_returns_finite_com",
      np.isfinite(res.COM_CDF) and np.isfinite(res.SNR_dB),
      "MLSE returned non-finite COM/SNR (COM_CDF=%s SNR_dB=%s)"
      % (res.COM_CDF, res.SNR_dB))

finish()
