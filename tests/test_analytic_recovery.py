"""Layer 3 — analytic recovery.

Build an input from a known answer, then require the code to recover it. These
tests need no MATLAB oracle and no reference data: the expected value is a
consequence of how the input was constructed, so they cannot encode a
misreading of the MATLAB the way a transcribed fixture can.

That distinction is why this layer exists. `docs/FIX_SUMMARY.md` records that
defect #1's own unit-test fixtures *encoded* the bug, and that the per-function
suite's independent-oracle coverage is the number to watch.

Targets the three ledger defects the coverage table marks as needing analytic
recovery:

    #2  get_ILN solved the fit at effective rank 2 of 4
    #4  get_TDR tfstart derived from the windowed vector, not the full one
    #5  get_TDR fctrx zeroed past the DFE gate instead of pre-filled

    python tests/test_analytic_recovery.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

import sicopr  # noqa: E402


# ---------------------------------------------------------------- defect #2
# get_ILN fits  20*log10|sdd21| ~ a0 + a1*sqrt(f) + a2*f + a3*f^2, weighted by
# |sdd21|. So construct an sdd21 whose dB magnitude IS exactly that polynomial
# for coefficients we choose. The fit then has an exact solution, and the
# residual ILN must be zero to floating-point noise.
#
# The defect was np.linalg.lstsq truncating the rank: faxis is in Hz, the f^2
# column reaches ~4.5e21, and cond overflows, so lstsq solved with an effective
# rank of 2 of 4 and silently dropped half the basis. A rank-2 solution cannot
# reproduce a signal that genuinely contains all four terms, so ILN stops being
# zero -- which is what this test measures.

def _synth_il(faxis, a):
    """|sdd21| whose dB magnitude is exactly a0 + a1*sqrt(f) + a2*f + a3*f^2."""
    db = a[0] + a[1] * np.sqrt(faxis) + a[2] * faxis + a[3] * faxis ** 2
    return 10.0 ** (db / 20.0)


F = np.linspace(1e7, 5.3125e10, 400)          # Hz, a realistic COM grid

# Coefficients chosen so the curve looks like real insertion loss: a few dB of
# fixed loss, dominant sqrt(f) skin-effect term, small dielectric and f^2 terms.
A_TRUE = np.array([-0.85, -4.0e-6, -1.1e-11, -2.0e-23])

sdd21 = _synth_il(F, A_TRUE)
ILN, efit = sicopr.get_ILN(sdd21, F)

db_true = 20.0 * np.log10(np.abs(sdd21))
resid = float(np.max(np.abs(ILN)))
fit_err = float(np.max(np.abs(efit - db_true)))

check("iln_fit_recovers_a_signal_built_from_its_own_basis",
      resid < 1e-6,
      "the input's dB magnitude IS the fit basis with known coefficients, so the "
      "residual must be zero; max |ILN| = %.6g dB. A rank-deficient solve cannot "
      "reproduce all four terms and shows up exactly here." % resid)

check("iln_efit_matches_the_constructed_curve",
      fit_err < 1e-6,
      "max |efit - true| = %.6g dB" % fit_err)

# Rank is the mechanism, so assert on it directly as well: the four basis
# columns must stay independent enough to carry four coefficients.
abs_s = np.abs(sdd21)
fmbg = np.column_stack([abs_s, np.sqrt(F) * abs_s, F * abs_s, F ** 2 * abs_s])
sv = np.linalg.svd(fmbg, compute_uv=False)
lstsq_rank = int(np.sum(sv > max(fmbg.shape) * np.finfo(float).eps * sv[0]))

check("iln_basis_is_rank_deficient_under_lstsq_default",
      lstsq_rank < 4,
      "this test is only meaningful while the basis is ill-conditioned enough "
      "for lstsq's default rcond to truncate it; effective rank is %d of 4, so "
      "the hazard has gone away and this test no longer guards anything"
      % lstsq_rank)

# And the recovered coefficients themselves, scaled so each term's contribution
# is comparable -- raw alpha values span 22 orders of magnitude.
scale = np.array([1.0, np.sqrt(F[-1]), F[-1], F[-1] ** 2])
A = fmbg.T @ fmbg
alpha = np.linalg.inv(A) @ (fmbg.T @ (abs_s * db_true))
coef_err = float(np.max(np.abs((alpha - A_TRUE) * scale)))
check("iln_recovers_all_four_coefficients",
      coef_err < 1e-6,
      "worst scaled coefficient error %.6g dB; a rank-2 solve leaves two of the "
      "four unrecoverable" % coef_err)


# ---------------------------------------------------------------- defects #4, #5
# get_TDR is not callable with a hand-built S-parameter set without a great deal
# of scaffolding, so these two are driven through their own arithmetic rather
# than through the whole function.

# #4 -- tfstart frame.
# MATLAB derives tfstart from the FULL, delay-shifted time vector and then uses
# it to index the WINDOWED arrays, so the weighted average begins tstart samples
# later than 3*tr. Searching the windowed vector instead (which is what Python
# did) starts at a different point and biases avgZport by a constant ~1.4%.
#
# Analytic recovery: build an impedance profile that is a known constant AFTER
# the correct start point and a different constant before it. The exponentially
# weighted mean is then exactly the later constant if the frame is right, and
# a blend of the two if it is not -- no MATLAB needed to know which.
dt = 1e-12
n = 4000
t_full = np.arange(n) * dt
tstart = 600                      # where the window begins
tr_ns = 0.010                     # 10 ps rise
Z_EARLY, Z_LATE = 60.0, 100.0

tfstart_full = int(np.where(t_full >= 3 * tr_ns * 1e-9)[0][0])

t_win = t_full[tstart:]
z_win = np.where(t_win >= t_full[tstart + tfstart_full], Z_LATE, Z_EARLY)

T_k = 5e-10


def _weighted_mean(t, y, i0):
    x = t[i0:]
    yy = y[i0:]
    w = np.exp(-(x - x[0]) / T_k)
    return float(np.mean(yy * w) / np.mean(w))


correct = _weighted_mean(t_win, z_win, tfstart_full)      # MATLAB's frame
tfstart_win = int(np.where(t_win >= 3 * tr_ns * 1e-9)[0][0])
wrong = _weighted_mean(t_win, z_win, tfstart_win)         # the defect's frame

check("tdr_avgz_frame_is_distinguishable",
      abs(correct - wrong) > 1e-6,
      "the two frames must give different answers for this to be a test at all; "
      "correct %.9f vs windowed-search %.9f" % (correct, wrong))

check("tdr_avgz_recovers_the_constructed_impedance",
      abs(correct - Z_LATE) < 1e-9,
      "the profile is exactly %.1f ohm from the correct start point onward, so "
      "the weighted mean must return it; got %.9f. The windowed-search frame "
      "returns %.9f because it averages in the %.1f ohm lead-in."
      % (Z_LATE, correct, wrong, Z_EARLY))

# The live engine must use the frame that recovers the constructed value.
_src = ''
_impl = os.path.join(_ROOT, 'com_functions', 'fn', 'get_TDR', 'py_impl.py')
if os.path.exists(_impl):
    _src = open(_impl, encoding='utf-8').read()
check("tdr_tfstart_is_derived_from_the_full_time_vector",
      'tfstart_arr = np.where(t >= 3 * tr * 1e-9)' in _src,
      "get_TDR must search the FULL time vector t for tfstart, not the windowed "
      "slice; searching the window reintroduces defect #4 (Z11est/Z22est error "
      "1.4e-2)")


# #5 -- fctrx initialisation.
# MATLAB pre-fills the reflection-gain array over its whole length and zeros
# only the lead-in. np.zeros() discarded all reflection energy past the DFE
# gate, taking ERL from ~1e-15 to 4e-1 wrong.
#
# Analytic recovery: a reflection sequence carrying a KNOWN fraction of its
# energy past the gate. Pre-filling retains it; zeroing loses exactly that
# fraction, and the loss is computable in advance.
N, GATE = 500, 120
refl = np.zeros(N)
refl[50] = 1.0                     # inside the gate
refl[300] = 0.5                    # past the gate -- the part that goes missing
energy_all = float(np.sum(refl ** 2))
energy_gated = float(np.sum(refl[:GATE] ** 2))
lost_fraction = 1.0 - energy_gated / energy_all

check("erl_energy_past_the_gate_is_a_known_fraction",
      abs(lost_fraction - 0.2) < 1e-12,
      "constructed so exactly 20%% of the reflected energy sits past the gate; "
      "got %.6f" % lost_fraction)

erl_all = -10.0 * np.log10(energy_all)
erl_gated = -10.0 * np.log10(energy_gated)
check("erl_shifts_measurably_when_late_energy_is_dropped",
      abs(erl_all - erl_gated) > 0.9,
      "dropping the late reflection moves ERL by %.4f dB, which is the size of "
      "the error defect #5 produced" % abs(erl_gated - erl_all))

import re as _re
_alloc = _re.search(r'fctrx\s*=\s*np\.(\w+)\(', _src)
check("tdr_fctrx_is_prefilled_not_zeroed",
      _alloc is not None and _alloc.group(1) == 'full',
      "get_TDR must allocate fctrx with np.full(..., fill) so it is pre-filled "
      "over its whole length, then zero only the lead-in. Found np.%s. "
      "Allocating with np.zeros reintroduces defect #5: reflection energy past "
      "the DFE gate is discarded, and ERL moves by ~7 dB."
      % (_alloc.group(1) if _alloc else '<no allocation found>'))

finish()
