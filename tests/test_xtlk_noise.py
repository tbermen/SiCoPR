"""Audit batch B13 spot-check: get_xtlk_noise crosstalk-noise ICN integral
(MATLAB lines 7974-8068 -> sicopr.py 11553).

Found by spot-checking a "version-identical" (4p14==4p15) function whose fn
test_verify.py passes: the upper bound of the ICN power sum is off by one.

  MATLAB 7976: index_f2 = find(faxis > fb, 1, 'first')   % 1-based
               ...  sum( ... PWF(1:index_f2) ... )        % INCLUDES first bin > fb
  sicopr.py:      index_f2 = argmax(f > fb)                  # 0-based
               ...  PWF[:index_f2] ...                    # EXCLUDES first bin > fb
  MATLAB empty case: index_f2 = length(faxis)   (all bins)
  sicopr.py empty case: len(f) - 1                 (omits the last bin)

Consequence: the crosstalk ICN sums (MDFEXT_ICN / MDNEXT_ICN, eq 93A-46/47)
omit one frequency bin near fb, slightly UNDER-estimating crosstalk noise ->
slightly optimistic COM whenever FEXT/NEXT aggressors are present.

This is B13-D18 (low-medium). Run: python tests/test_xtlk_noise.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import sicopr  # noqa: E402

TOL = 1e-12


def run_case(faxis, fb):
    """Build a minimal 1-aggressor FEXT chdata and return sicopr.py's MDFEXT_ICN
    plus the MATLAB-correct reference (inclusive upper bound)."""
    f = np.asarray(faxis, dtype=float)
    n = len(f)
    sdd21 = 0.5 * np.ones(n)          # flat crosstalk transfer
    param = SimpleNamespace(fb=fb, samples_per_ui=32,
                            sample_dt=1.0 / (32 * fb), f2=fb, levels=4,
                            RxFFE_cmx=2, RxFFE_cpx=2)
    ch0 = SimpleNamespace(faxis=f)
    ch1 = SimpleNamespace(type='FEXT', sdd21ctf=sdd21, delta_f=(f[1] - f[0]), A=1.0)
    chdata = [ch0, ch1]
    # sicopr.py result (FEXT single-aggressor). upsampled_txffe=[0] -> PWF_tx=ones; C=None.
    sigma_fext_py = sicopr.get_xtlk_noise(np.array([0.0]), 'FEXT', param, chdata)

    # Independent MATLAB-correct reference: index_f2 is the 1-based find value,
    # used directly as the Python exclusive end (== inclusive MATLAB end).
    mask = f > fb
    idx_ml = (int(np.argmax(mask)) + 1) if np.any(mask) else n   # 1-based find / all
    temp_angle = param.samples_per_ui * param.sample_dt * np.pi * f
    if f[0] == 0.0:
        temp_angle[0] = 1e-20
    SINC = np.sin(temp_angle) / temp_angle
    PWF = SINC ** 2
    MDFEXT = np.abs(sdd21)
    ref_ICN = float(np.sqrt(2 * (f[1] - f[0]) / fb *
                            np.sum(1.0 ** 2 * PWF[:idx_ml] * MDFEXT[:idx_ml] ** 2)))
    scale = float(np.sqrt((param.levels ** 2 - 1) / (3 * (param.levels - 1) ** 2)))
    ref_sigma = ref_ICN * scale
    return float(sigma_fext_py), ref_sigma, idx_ml


# Case A: fb inside the band, so there IS a first bin above fb (index arithmetic
# diverges by one). faxis 0..40 GHz step 1 GHz, fb = 20 GHz -> first bin > fb at
# 0-based index 21; MATLAB includes it, sicopr.py omits it.
faxis = np.arange(0.0, 40e9 + 1e9, 1e9)
py_a, ref_a, idx_a = run_case(faxis, 20e9)
check("xtlk_index_f2_has_bin_above_fb", idx_a == 22,
      "expected MATLAB 1-based index_f2 == 22 (first bin >20GHz at 0-based 21)")
# EXPECTED FAIL: sicopr.py omits the first bin above fb -> differs from MATLAB ref.
xcheck("xtlk_MDFEXT_ICN_matches_matlab_upper_bound",
      abs(py_a - ref_a) <= TOL,
      "DIVERGENT (B13-D18, low-med): sicopr.py FEXT sigma=%.9e but MATLAB-correct "
      "(inclusive upper bound, index_f2=22) =%.9e. sicopr.py index_f2=argmax(f>fb) "
      "omits the first bin above fb in the ICN sum (sicopr.py 11553; MATLAB 7976/8051)."
      % (py_a, ref_a))
# Confirm the gap equals exactly the omitted index-21 term (mechanism).
f = faxis
ta = 32 * (1.0 / (32 * 20e9)) * np.pi * f
ta[0] = 1e-20 if f[0] == 0 else ta[0]
sinc = np.sin(ta) / ta
scale = float(np.sqrt((4 ** 2 - 1) / (3 * (4 - 1) ** 2)))
omitted_term = 2 * 1e9 / 20e9 * (1.0 ** 2 * (sinc[21] ** 2) * (0.5 ** 2))
ref_sq = (ref_a / scale) ** 2
py_sq = (py_a / scale) ** 2
check("xtlk_gap_equals_first_bin_above_fb",
      abs((ref_sq - py_sq) - omitted_term) <= 1e-9 * max(1.0, ref_sq),
      "the sicopr.py/MATLAB ICN^2 gap (%.6e) does not equal the omitted first-bin "
      "term (%.6e)" % (ref_sq - py_sq, omitted_term))

# Case B: no bin above fb (fb above the whole grid) -> MATLAB uses all N bins,
# sicopr.py uses len-1 (omits the last bin). Another off-by-one.
faxis_b = np.arange(0.0, 20e9 + 1e9, 1e9)   # 0..20 GHz
py_b, ref_b, idx_b = run_case(faxis_b, 25e9)   # fb above grid -> no f>fb
check("xtlk_index_f2_empty_case_uses_all_bins", idx_b == len(faxis_b),
      "empty case reference should use all %d bins" % len(faxis_b))
xcheck("xtlk_MDFEXT_ICN_empty_case_matches_matlab",
      abs(py_b - ref_b) <= TOL,
      "DIVERGENT (B13-D18, low-med): empty-case sicopr.py FEXT sigma=%.9e but "
      "MATLAB (all bins) =%.9e. sicopr.py uses len(f)-1, omitting the last bin "
      "(sicopr.py 11553)." % (py_b, ref_b))

finish()
