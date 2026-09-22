"""Audit batch B15 spot-check: MLSE Gaussian SNR_DFE_eqivalent (MATLAB 2271-2352
-> sicopr.py 2713-2783).

B15-D19 (low, diagnostic-only): in the Gaussian equivalent-SNR step, MATLAB
line 2311 has
    qfunc( (1-2*alpha)*main/(L-1)*sigma_noise )
which, by MATLAB's left-associative */ precedence, puts sigma_noise in the
NUMERATOR: (1-2*alpha)*main*sigma_noise/(L-1). sicopr.py writes
    _MLSE__qfunc((1-2*alpha)*A_peak/((L-1)*sigma_noise))
with sigma_noise in the DENOMINATOR. (The commented-out reference at MATLAB 2295
divides, so MATLAB 2311 looks like a typo - but the audit reproduces MATLAB.)

Impact: only the reported diagnostics SNR_DFE_eqivalent_Gaussian and
delta_com_Gaussian differ. The operative COM is unaffected: MATLAB sets BOTH
COM_Gaussian and COM_CDF to new_com_CDF (the CDF path, 2345-2346), and sicopr.py
does the same (2778-2779). So no COM value changes.

Run: python tests/test_mlse.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import sicopr  # noqa: E402


# --- 1. Root-cause: the two qfunc-argument forms differ (precedence) ----------
alpha, main, L, sigma = 0.3, 0.1, 4, 0.02
ml_arg = (1 - 2 * alpha) * main / (L - 1) * sigma          # MATLAB 2311 (left-assoc)
py_arg = (1 - 2 * alpha) * main / ((L - 1) * sigma)         # sicopr.py form
# B15-D19. The two forms differ algebraically by sigma^2 -- that is the check
# below, and it is a statement about arithmetic, not about the engine.
#
# The previous version of this file recorded the divergence with an xcheck on
# `abs(ml_arg - py_arg) <= 1e-15`, where BOTH sides were hand-written here from
# invented values (alpha=0.3, main=0.1, L=4, sigma=0.02). It never called
# sicopr, so it reported "divergent" unconditionally and would have gone on
# doing so after any fix -- the same fault as the s21 grid-round record in
# tests/test_impulse_spectrum.py and the triple-transit record in
# tests/test_optimizer_fom.py, both since corrected.
#
# What IS measurable: this expression feeds SNR_DFE_eqivalent -> delta_com ->
# the reported delta_COM column, and delta_COM agrees with the MATLAB reference
# to 15 significant digits on all 208 correlation cases (e.g. 1.371137901447263
# vs 1.37113790144726). So whatever the precedence reading, the divergence is
# not observable in any reported output on that corpus. It is NOT therefore
# proven absent -- the corpus is one channel family and the agreement may not
# generalise -- so the form the engine uses is pinned here instead of the
# divergence being closed.
import inspect as _inspect
_mlse_src = _inspect.getsource(sicopr.MLSE)
_arg_lines = [l.strip() for l in _mlse_src.splitlines()
              if 'qfunc' in l and '1 - 2 * alpha' in l]
check("mlse_gaussian_qfunc_arg_form_is_pinned",
      any('/ (L - 1) * sigma_noise' in l for l in _arg_lines),
      "the engine's Gaussian q-function argument changed form. It must "
      "MULTIPLY by sigma_noise: both 4p16p0 and the 4p15p0 adaptive-local-search "
      "build read `(1-2*alpha)*main/(L-1)*sigma_noise`, and MATLAB's "
      "left-associative precedence makes that ((1-2*alpha)*main/(L-1))*sigma_noise. "
      "Until 2026-09-22 this check pinned the DIVIDE form instead, as a deliberate "
      "deviation: dividing is the physically sensible reading, since the multiply "
      "sends qfunc to 0.5 rather than 0 as noise vanishes. The reference says "
      "otherwise, and the standing rule is to match it and raise the oddity with "
      "the COM ad hoc, so the pin was flipped. This surfaces in delta_COM: the "
      "208-case correlation must confirm it. Found: %s"
      % (_arg_lines or '<no matching line>'))
check("mlse_gaussian_arg_ratio_is_sigma_squared",
      abs((ml_arg / py_arg) - sigma ** 2) <= 1e-12,
      "the MATLAB/sicopr.py arg ratio is not sigma^2 (mechanism check)")


# --- 2. MLSE runs and the operative COM is diagnostic-independent -------------
# Build a symmetric noise PDF via the audited normal_dist, and a CDF (cumsum).
pdf = sicopr.normal_dist(0.02, 7, 1e-4)          # sigma=0.02
cdf = np.cumsum(np.asarray(pdf.y, dtype=float))
param = SimpleNamespace(levels=4, specBER=1e-5)
res = sicopr.MLSE(param, 0.3, 0.10, 0.05, pdf, cdf)   # A_s >= A_ni -> MLSE applied

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
