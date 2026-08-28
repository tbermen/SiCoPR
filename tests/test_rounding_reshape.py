"""Audit batch B10: G8 rounding + G9 reshape/flatten order, cross-cutting scans.

Scope (audit prompt section 3 items 8 and 9):
  G8: MATLAB round() is half-away-from-zero; NumPy/Python round is banker's
      (round-half-to-even). Verify every _mround helper implements half-away, and
      characterise the sites that use a bare round.
  G9: MATLAB is column-major; NumPy defaults to row-major. Verify the one
      order-sensitive reshape uses order='F' and that the vector-coercion ravels
      are order-agnostic.

Findings from the static scan (sicopr.py):
  - 6 _mround helpers (half-away), used at the port-order quarter/three-quarter
    slice sites and the adaptive-search radius - where an off-by-one changes the
    selected S-parameter sub-band or search radius.
  - ~50 other rounding sites use bare round/np.round (banker's): PDF bin snapping
    (d_cpdf/Init_PDF_Fast/get_pdf_from_sampled_signal), grid-length counts
    (s21_to_impulse_DC = B03-D7), normal_dist Min, nui=round(len/M),
    combine_pdf shift. These differ from MATLAB only for exact half-integer
    inputs (measure-zero for continuous data; integer Min-sums are no-ops).
  - reshape order: exactly one order-sensitive reshape (get_PSDs fold) uses
    order='F'; get_pdf/get_pdf_full vs-matrices use reshape (MATLAB
    reshape+transpose == numpy C-order); no .flatten() anywhere; the ~398 ravels
    are 1-D coercion.

Run: python tests/test_rounding_reshape.py
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import sicopr  # noqa: E402


# ===========================================================================
# 1. G8: every _mround helper is half-away-from-zero (ML round semantics)
# ===========================================================================
mround_helpers = {
    '_OptFom_Adaptive_Local_Search__mround': sicopr._OptFom_Adaptive_Local_Search__mround,
    '_auto_port_order__mround': sicopr._auto_port_order__mround,
    '_compute_hard_cap__mround': sicopr._compute_hard_cap__mround,
    '_read_Nport_touchstone__mround': sicopr._read_Nport_touchstone__mround,
    '_read_p4_s4params__mround': sicopr._read_p4_s4params__mround,
    '_read_s4p_files__mround': sicopr._read_s4p_files__mround,
}
# MATLAB round: half away from zero.
cases = [(0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4),
         (-0.5, -1), (-1.5, -2), (-2.5, -3),
         (0.4, 0), (0.6, 1), (-0.6, -1), (10.5, 11)]
for name, fn in mround_helpers.items():
    ok = all(int(fn(x)) == exp for x, exp in cases)
    got = [int(fn(x)) for x, _ in cases]
    check("mround_half_away_%s" % name, ok,
          "%s not half-away: got %s expected %s" % (name, got, [e for _, e in cases]))

# Contrast: Python builtin/np round is banker's on the .5 cases (documents the
# reason the _mround helpers exist).
check("python_round_is_bankers_reference",
      round(2.5) == 2 and round(0.5) == 0 and int(np.round(2.5)) == 2,
      "Python/np round unexpectedly not banker's")

# ===========================================================================
# 2. G8: a real bare-round site (normal_dist Min) uses banker's, not half-away
#    -> EXPECTED FAIL documenting B10-D14 at a concrete call site.
# ===========================================================================
# Choose 2*nsigma*sigma/binsize = 2.5 exactly: nsigma=1, sigma=1.25, binsize=1.
nd = sicopr.normal_dist(1.25, 1, 1.0)
# MATLAB: Min = -round(2.5) = -3 (half away). Python: -round(2.5) = -2 (banker's).
xcheck("normal_dist_Min_uses_matlab_half_away",
      nd.Min == -3,
      "DIVERGENT (B10-D14, low): normal_dist Min=%d; MATLAB round(2.5)=3 gives "
      "-3 but Python banker's round(2.5)=2 gives -2. Bare round at a PDF-axis "
      "site (py 12412). Only bites exact half-integer 2*nsigma*sigma/binsize."
      % nd.Min)
# Confirm it IS the banker's result (positive confirmation of the mechanism).
check("normal_dist_Min_is_bankers",
      nd.Min == -2,
      "normal_dist Min is neither the MATLAB nor the banker's value: %d" % nd.Min)

# nui = round(len/M): banker's vs half-away at len/M = 2.5 (e.g. len=80, M=32).
check("nui_round_len_over_M_is_bankers",
      round(80 / 32) == 2,
      "round(80/32)=round(2.5) banker's expected 2; MATLAB half-away gives 3 "
      "(get_pdf/get_pdf_full nui, py 10916/11284)")

# ===========================================================================
# 3. G9: the get_PSDs fold is the one order-sensitive reshape and uses order='F'
# ===========================================================================
num_ui, M = 6, 4
v = np.arange(num_ui * M, dtype=float)            # 0..23
# MATLAB: sum(reshape(v, num_ui, M).')  (column-major reshape, transpose, sum cols)
# reference computed index-wise (order-independent transcription):
ref = np.zeros(num_ui)
for k in range(num_ui):
    ref[k] = sum(v[k + num_ui * m] for m in range(M))   # v[k], v[k+num_ui], ...
py_fold = sicopr._get_PSDs__fold_psd(v, num_ui, M)
check("get_PSDs_fold_uses_column_major_orderF",
      np.max(np.abs(py_fold - ref)) <= 1e-12,
      "fold_psd (order='F') mismatch vs column-major reference")
# Demonstrate that a naive C-order fold would give the WRONG answer (why 'F' matters).
c_order_wrong = v.reshape(num_ui, M).sum(axis=1)   # row-major grouping
check("get_PSDs_fold_C_order_would_be_wrong",
      np.max(np.abs(c_order_wrong - ref)) > 1e-6,
      "C-order fold coincidentally equals column-major (not a discriminating case)")

# ===========================================================================
# 4. G9: vector-coercion reshapes/ravels are order-agnostic
# ===========================================================================
# reshape(-1,1) and reshape(1,-1) on a 1-D array give the same result for C or F.
a = np.array([1.0, 2.0, 3.0, 4.0])
check("reshape_col_row_vector_order_agnostic",
      np.array_equal(a.reshape(-1, 1), a.reshape(-1, 1, order='F'))
      and np.array_equal(a.reshape(1, -1), a.reshape(1, -1, order='F'))
      and np.array_equal(a.ravel(), a.ravel(order='F')),
      "1-D vector coercion is not order-agnostic (unexpected)")

# The get_pdf_full vs reshape equals MATLAB reshape(v,samp_UI,nrows).' (transpose).
samp_UI, n_rows = 4, 5
vv = np.arange(samp_UI * n_rows, dtype=float)
py_vs = vv.reshape(n_rows, samp_UI)                       # sicopr.py get_pdf_full path
ml_vs = vv.reshape(samp_UI, n_rows, order='F').T          # MATLAB reshape+transpose
check("get_pdf_full_reshape_equals_matlab_transpose",
      np.array_equal(py_vs, ml_vs),
      "C-order reshape != MATLAB column-major reshape+transpose")

finish()
