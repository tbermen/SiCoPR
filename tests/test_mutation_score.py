"""Layer 5 -- are the tests awake?

Every other gate in this directory asks whether the inventory is COMPLETE. This
one asks a different question that none of them can answer: whether the tests
behind it are AWAKE. A row can be covered by every instrument we have and still
be closed by a test that catches nothing.

`com_functions/verification/mutations.py` reintroduces a defect that actually
happened in this port, runs that function's own test, and records whether the
test failed. A test that still passes did not discriminate.

3 of the 6 `np.std(..., ddof=1)` sites SURVIVE -- the very defect that prompted
the verification contract. It was fixed in d5bff6c, and the suite would still
not notice it coming back at interp_Sparam lines 205 and 249 or
s21_to_impulse_DC line 182. That is the value of asking this question by
computation instead of from memory.

(The first run of this gate reported 4 of 6, and was wrong. It was measuring
stale bytecode: see the note in `mutations.run_test`. The number above is from
the corrected engine. A tool that measures whether tests lie has no business
lying itself, so both the bug and the wrong number are recorded rather than
quietly replaced.)

## Why this is a pinned set and not a score

A bare percentage lets a new sleeping test hide behind a fixed one. The set
names each gap as `operator:function`, so a NEW survivor fails by name, and a
survivor that gains a real test must be deleted from the set, which holds the
gain. Same contract as KNOWN_UNCOVERED and the BASELINE_ ratchets.

## What this proves, and what it does not

Catching a mutant proves the test is NOT VACUOUS. It does NOT prove the test is
correct, and it does not prove the port is right. A test written from a READING
of the MATLAB pins whatever the port already does, so mutation confirms the test
is awake while test and port are wrong together. Only executing the reference
separates those, which is why the Octave oracle is mandatory and not merely
preferred. Do not quote this gate as evidence of correctness.

First run is a few minutes; results are cached by the content hash of BOTH the
implementation and its bound test, so an unchanged pair is not re-run. Change
either and it re-runs.

    python tests/test_mutation_score.py
    python com_functions/verification/mutations.py --list
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish              # noqa: E402
from com_functions.verification import mutations   # noqa: E402

# Pinned 2026-09-22 from the catalogue's first full run. Each entry is a
# function whose own test does not notice a defect class present in its code.
# Remove an entry when its test learns to catch it; never add one to make a red
# run green. An entry here is a REAL GAP, not an accepted divergence: unlike an
# xcheck there is no argument that the gap is correct, only that it is known.
KNOWN_SURVIVORS = frozenset([
    'drop_dot_copy:Bathtub_Contribution_Wrapper',
    'drop_dot_copy:COM_FD_to_TD',
    'drop_dot_copy:COM_eye_width',
    'drop_dot_copy:OptFom_Compute_DFE',
    'drop_dot_copy:OptFom_Compute_TXFFE',
    'drop_dot_copy:OptFom_Create_Output',
    'drop_dot_copy:OptFom_Update_BEST_Post_Optimize',
    'drop_dot_copy:OptFom_Update_Best_Settings_EQ_Failed',
    'drop_dot_copy:OptFom_Update_Best_Setttings',
    'drop_dot_copy:SNDR_ref',
    'drop_dot_copy:add_pkg_with_die',
    'drop_dot_copy:applyDFEbk',
    'drop_dot_copy:auto_port_order',
    'drop_dot_copy:comb_fct',
    'drop_dot_copy:combine_pdf_same_voltage_axis',
    'drop_dot_copy:dfe_clipper',
    'drop_dot_copy:floatingDFE',
    'drop_dot_copy:get_PSDs',
    'drop_dot_copy:get_TDR',
    'drop_dot_copy:interp_Sparam',
    'drop_dot_copy:make_full_pkg',
    'drop_dot_copy:make_pkg',
    'drop_dot_copy:optimize_fom',
    'drop_dot_copy:plot_bathtub_curves',
    'drop_dot_copy:process_sxp',
    'drop_dot_copy:read_Nport_touchstone',
    'drop_dot_copy:read_ParamConfigFile',
    'drop_dot_copy:read_s4p_files',
    'drop_dot_copy:s21_pkg',
    'drop_dot_copy:s21_to_impulse_DC',
    'drop_dot_copy:synth_tline',
    'mlength_to_len:get_RILN_cmp_td',
    'mmax_to_np_max:Burst_Probability_Calc',
    'mmax_to_np_max:COM_eye_width',
    'mmax_to_np_max:Create_Noise_PDF',
    'mmax_to_np_max:OptFom_Compute_CTLE',
    'mmax_to_np_max:OptFom_Create_Output',
    'mmax_to_np_max:OptFom_Plot_Best_Results',
    'mmax_to_np_max:OptFom_Update_Best_Settings_EQ_Failed',
    'mmax_to_np_max:Output_Arg_Fill',
    'mmax_to_np_max:RILN_TD',
    'mmax_to_np_max:adjust_Rx_noise_for_quantization',
    'mmax_to_np_max:get_ILN_cmp_td',
    'mmax_to_np_max:get_cm_noise',
    'mmax_to_np_max:get_pdf',
    'mmax_to_np_max:get_xtlk_noise',
    'mmax_to_np_max:parameter_size_adjustment',
    'mmax_to_np_max:read_s4p_files',
    'mmax_to_np_max:s21_to_impulse_DC',
    'mmin_to_np_min:COM_eye_width',
    'mmin_to_np_min:MMSE',
    'mmin_to_np_min:findbankloc',
    'mmin_to_np_min:force',
    'mmin_to_np_min:get_cm_noise',
    'mmin_to_np_min:plot_modal',
    'mround_to_np_round:OptFom_Adaptive_Local_Search',
    'mround_to_np_round:calculate_delay_CausalityEnforcement',
    'mround_to_np_round:get_ILN_cmp_td',
    'mround_to_np_round:optimize_fom',
    'ne_zero_to_gt_zero:COM_eye_width',
    'ne_zero_to_gt_zero:Create_Noise_PDF',
    'ne_zero_to_gt_zero:MLSE_U1_c_178A',
    'ne_zero_to_gt_zero:MMSE',
    'ne_zero_to_gt_zero:MMSE_FOM',
    'ne_zero_to_gt_zero:OptFom_Compute_DFE',
    'ne_zero_to_gt_zero:OptFom_Plot_Best_Results',
    'ne_zero_to_gt_zero:Output_Arg_Fill',
    'ne_zero_to_gt_zero:RXFFE_Illegal',
    'ne_zero_to_gt_zero:Tx_FFE_Filter',
    'ne_zero_to_gt_zero:applyDFEbk',
    'ne_zero_to_gt_zero:calculate_delay_CausalityEnforcement',
    'ne_zero_to_gt_zero:capture_RIL_RILN',
    'ne_zero_to_gt_zero:com_ieee8023_',
    'ne_zero_to_gt_zero:floatingDFE',
    'ne_zero_to_gt_zero:force',
    'ne_zero_to_gt_zero:get_ILN_cmp_td',
    'ne_zero_to_gt_zero:get_PSDs',
    'ne_zero_to_gt_zero:get_TDR',
    'ne_zero_to_gt_zero:get_pdf',
    'ne_zero_to_gt_zero:get_pdf_full',
    'ne_zero_to_gt_zero:get_sigma_eta_ACCM_noise',
    'ne_zero_to_gt_zero:get_sigma_noise',
    'ne_zero_to_gt_zero:interp_Sparam',
    'ne_zero_to_gt_zero:optimize_fom',
    'ne_zero_to_gt_zero:read_ParamConfigFile',
    'ne_zero_to_gt_zero:read_s4p_files',
    'ne_zero_to_gt_zero:s21_to_impulse_DC',
    'solve_to_lstsq:MMSE',
    'solve_to_lstsq:MMSE_FOM',
    'solve_to_lstsq:force',
    'solve_to_lstsq:read_Nport_touchstone',
    'solve_to_lstsq:read_p4_s4params',
    'solve_to_lstsq:read_s4p_files',
    'std_ddof:interp_Sparam',
    'std_ddof:s21_to_impulse_DC',
])

# Mutants that provably change nothing. They read as "not caught" and are not
# test failures. Decided state, and falsifiable: if the mutant ever stops being
# equivalent a test starts catching it and the entry is contradicted below.
#
# Keyed `operator:function:line`, NOT `operator:function`. A whole function is
# far too coarse: MMSE has several .copy() sites and only ONE of them is
# provably redundant, so a function-level entry would excuse the other five.
# Line numbers move when a file is edited, and that is the safe direction --
# a stale entry stops matching, the site reappears as a survivor, and the gate
# says so rather than staying quiet.
#
# The argument must be that the mutant CANNOT change the result. "No test
# happens to cover this input" is a reason to write a test, and belongs in
# KNOWN_SURVIVORS as a real gap. See verification/equivalent_mutants.md.
EQUIVALENT = frozenset([
    'drop_dot_copy:MMSE:371',
    'drop_dot_copy:get_PSDs:436',
    'drop_dot_copy:get_pdf_full:128',
])


def main():
    rows = mutations.evaluate()

    # Filter equivalent mutants per SITE, before collapsing to operator:
    # function, so excusing one redundant copy does not excuse the others in
    # the same function.
    def site(r):
        return '%s:%s:%d' % (r['op'], r['fn'], r['line'])

    real = [r for r in rows if site(r) not in EQUIVALENT]
    survived = sorted({'%s:%s' % (r['op'], r['fn'])
                       for r in real if r['outcome'] == 'survived'})

    matched = {site(r) for r in rows}
    orphan = sorted(EQUIVALENT - matched)
    check('every_equivalent_mutant_entry_still_matches_a_site',
          not orphan,
          'these EQUIVALENT entries match no mutation site any more, so the '
          'argument they record is about code that has moved or gone. '
          'Re-check each and update the line, or delete it: %s' % orphan)

    bugs = ['%s:%s line %d' % (r['op'], r['fn'], r['line']) for r in rows
            if r['outcome'] == 'catalogue_bug_mutant_does_not_parse']
    failing = sorted({r['fn'] for r in rows
                      if r['outcome'] == 'test_already_failing'})

    caught = sum(1 for r in rows if r['outcome'] == 'caught')
    total = len(rows)

    print('\n%d mutants, %d caught, %d survived, %d equivalent-mutant entries'
          % (total, caught, len(survived), len(EQUIVALENT)))

    # A mutant that does not parse fails the test for a reason that has nothing
    # to do with the defect, so it would score as a catch and certify the test
    # on evidence of nothing. It is a bug in the OPERATOR and must be loud.
    check('every_mutant_is_syntactically_valid',
          not bugs,
          'these mutants do not parse, so they prove nothing about the test '
          'that "caught" them. Fix the operator in mutations.py: %s' % bugs)

    # If a test was already red the mutant is scored against a broken baseline.
    check('every_bound_test_passes_before_mutation',
          not failing,
          'these functions\' tests fail on the UNMUTATED tree, so no mutation '
          'result for them means anything: %s' % failing)

    new = sorted(set(survived) - KNOWN_SURVIVORS)
    check('no_new_sleeping_test',
          not new,
          'a defect of a class that has really happened in this port was '
          'introduced into these functions and their own tests still passed. '
          'Either strengthen the test or, if the mutant provably changes '
          'nothing, add it to EQUIVALENT with the reason: %s' % new)

    fixed = sorted(KNOWN_SURVIVORS - set(survived))
    check('known_survivor_list_is_current',
          not fixed,
          'these now catch their mutant -- delete them from KNOWN_SURVIVORS so '
          'the gain is held and cannot silently regress: %s' % fixed)

    finish()


if __name__ == '__main__':
    main()
