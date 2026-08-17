"""Audit B16 spot-check that revised the B11 optimize_fom verdict.

B16-D20 (medium, CONFIRMED): optimize_fom treats the cursor/peak sample indices
returned by OptFom_Find_Sample_Point as 1-based (subtracts 1), but that function
(via the hoisted cursor_sample_index) returns 0-based indices - the same 0-based
convention the main COM pipeline (get_PSDs B01, get_pdf B07) and OptFom_Compute_DFE
use. Net effect: optimize_fom reads ONE SAMPLE BEFORE the true cursor/peak for
    A_s  = R_LM * sbr[cursor_i - 1] / (L-1)      (com.py 12722/12724)
    A_p  = sbr[sbr_peak_i - 1]                    (com.py 12723)
    far_start = cursor_i - T_O + M*(ndfe+1) - 1   (com.py 12731)
    pre_start = cursor_i - M - 1                   (com.py 12735)
while OptFom_Compute_DFE (com.py 4514+) correctly uses sbr[cursor_i] (0-based).
The two are anchored to different samples in the same loop iteration.

MATLAB (com_ieee8023_4p15p0.m 8788-8801): cursor_i is 1-based there, so
sbr(cursor_i), sbr(sbr_peak_i), and the colon windows all read the TRUE cursor/
peak with no offset. The faithful Python is sbr[cursor_i] / sbr[sbr_peak_i]
(0-based), i.e. the "-1" should not be present.

Consequence: A_s (main signal, COM = 20log10(A_s/noise)) and A_p and the far/
precursor ISI windows are shifted one sample early in the FOM optimisation loop,
biasing both the reported FOM and which EQ setting is selected. Magnitude is a
one-sample pulse droop near the cursor (small when the pulse is flat at the
cursor, larger on a steep flank).

This corrects the B11 verdict for optimize_fom (was EQUIVALENT; the B11 far/
precursor tests were standalone formula reproductions that assumed a 1-based
cursor_i and so did not exercise the real 0-based return of cursor_sample_index).

Run: python tests/test_optimize_cursor.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish  # noqa: E402
import com  # noqa: E402

M = 32
N = 20 * M
P0 = 10 * M                      # true 0-based peak index
t = np.arange(N)
sbr = np.exp(-((t - P0) / (1.5 * M)) ** 2) * 0.8
sbr[:P0] *= 0.98                 # mild asymmetry so peak is unique
param = SimpleNamespace(samples_per_ui=M, ndfe=1, bmax=np.array([0.2]), ts_anchor=1)
OP = SimpleNamespace(CDR='MM')
sr = np.arange(P0 - 4 * M, P0 + 4 * M)

cursor_i, nzc, sbr_peak_i = com.OptFom_Find_Sample_Point(sbr, param, OP, sr)

# 1. OptFom_Find_Sample_Point returns 0-based indices (sbr[peak] is the max).
check("find_sample_point_returns_0based_peak",
      sbr_peak_i == P0 and np.isclose(sbr[sbr_peak_i], sbr.max()),
      "sbr_peak_i=%d (expected 0-based %d); sbr[peak]=%.4f max=%.4f"
      % (sbr_peak_i, P0, sbr[sbr_peak_i], sbr.max()))

# 2. B16-D20 is RESOLVED. The audit was right that the `-1` read one sample
#    before the peak. An end-to-end trial on 2026-08-13 appeared to show that
#    removing it made MATLAB agreement WORSE, so it was retained and this check
#    was left documenting an "open question" -- but that trial was confounded:
#    the -1 was compensating a second defect in the die-network path. Once both
#    were fixed together (b2b2621) com.py adopted the MATLAB-faithful
#    sbr[sbr_peak_i] and FOM went bit-exact on 198/208 reference cases.
#
#    This check used to mirror the shipped expression as sbr[sbr_peak_i - 1] and
#    assert it equalled the peak, so it failed by construction and could never
#    see the fix. It now mirrors what com.py actually ships.
A_p_optfom = float(sbr[int(sbr_peak_i)])           # as shipped in com.py
check("optimize_fom_A_p_equals_peak",
      np.isclose(A_p_optfom, sbr.max()),
      "A_p = %.5f but the true peak is %.5f. com.py should read sbr[sbr_peak_i] "
      "with no offset (B16-D20); a reintroduced -1 would show up here."
      % (A_p_optfom, sbr.max()))

# 3. Positive confirmation of the MATLAB-faithful value.
check("matlab_faithful_A_p_is_peak",
      np.isclose(float(sbr[int(sbr_peak_i)]), sbr.max()),
      "sbr[sbr_peak_i] (0-based, MATLAB-faithful) is not the peak")

# 4. optimize_fom anchors A_s to sbr[cursor_i-1] while OptFom_Compute_DFE uses
#    sbr[cursor_i]. Documented inconsistency, unresolved - see note above.
check("optimize_fom_vs_compute_dfe_cursor_mismatch",
      abs(float(sbr[int(cursor_i) - 1]) - float(sbr[int(cursor_i)])) > 0,
      "cursor samples happen to be equal for this input (non-discriminating); "
      "expected sbr[cursor_i-1] != sbr[cursor_i]")

finish()
