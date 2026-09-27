"""MATLAB-anchored stage oracles: real reference values, no ask on anyone.

Every one of the 156 per-function test files asserts what Python already does.
None pins a MATLAB-produced number. That is the structural reason the eight
engine defects survived 876 unit tests: the suite could not tell "correct" from
"consistently wrong".

The ideal fix -- a per-function dump from the MATLAB side -- is a large favour
to ask of someone who gets nothing back from it. This gets most of the way
there from the two MATLAB reference workbooks, which carry ~260 columns of
genuine MATLAB output per case, many of them the output of one identifiable
stage. Neither they nor anything derived from them is distributed here.

A local extractor -- not part of the published repository -- distils those into
an oracle file of all 208 reference cases, 35 scalar quantities plus 14 VECTOR
families. It is not tracked: point COM_STAGE_ORACLES at a local copy, or this
test skips. This file checks it.

The vectors matter more than the scalars: a tap set is wrong in ways a gain is
not. RxFFE is 87 taps, TXLE_taps up to 4, floating_tap_locations 8, and the
package families (Pkg_len_*, pkg_Z_c, C_diepad, L_comp) are MATLAB's record of
the package it actually built -- Pkg_len_RX alone would have caught engine
defect #1, where only the TX z_p was transposed so the RX package came from a
matrix row.

Two workbook quirks worth knowing, both handled:
  - C_diepad, L_comp and C_bump are stored as TEXT ('4e-14'), so a plain
    isinstance(v, (int, float)) test drops those families silently.
  - The wXtalk workbook writes TXLE_taps_1..4 (the full vector); the
    woXtalk one writes a single TXLE_taps column. Same answer at different granularity, not a
    disagreement about which taps were used.

Two scalars are legitimately sparse: sgm_rjit 160/208 and sgm_xt 203/208, blank
where the case has no such term.

Three layers, cheapest first:

  1. Structure. The oracle file covers the stages and functions it claims, and
     every function it names still exists. A MATLAB update that renames or
     removes a function fails here in milliseconds.
  2. Provenance. The MATLAB values in the oracle agree with the independently
     produced comparison table (report_data/compare.csv). This is what proves
     the extractor read the RIGHT columns -- the workbooks repeat several
     header names, and taking the wrong duplicate produced fake 2.58 dB
     discrepancies during the correlation before it was caught.
  3. Live. With channel data present and COM_ORACLE_LIVE=1, run the engine and
     compare its output to the MATLAB oracle case by case. Skipped by default
     because it needs the gitignored 210 MB channel set and minutes per case.

Run: python tests/test_matlab_stage_oracles.py
     COM_ORACLE_LIVE=1 python tests/test_matlab_stage_oracles.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import collections as _collections
import csv
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

# The oracle lives outside the repository because it is derived from the MATLAB
# reference workbooks. COM_STAGE_ORACLES points at the local copy, so the same
# test that SKIPS in a fresh clone runs for real on a machine that holds the
# data. Without it, ~800 KB of genuine MATLAB output that is already on disk
# sits unused and this file contributes zero checks -- which is how it stood
# until 2026-09-22.
ORACLE = os.environ.get('COM_STAGE_ORACLES') or os.path.join(
    _HERE, 'oracles', 'matlab_stage_oracles.json')
COMPARE = os.path.join(_ROOT, 'report_data', 'compare.csv')
REGISTRY = os.path.join(_ROOT, 'com_functions', 'registry.json')

# --------------------------------------------------------------- layer 1
# The oracle is DERIVED FROM the MATLAB reference workbooks -- it is a
# distillation of the same numbers -- so it is not redistributable and is not in
# the repository. Absent is the normal state of a fresh clone: skip, do not fail.
# Present-but-wrong is still a failure, which is what everything below tests.
if not os.path.exists(ORACLE):
    print("SKIP matlab_stage_oracles: %s not present. It is generated from the "
          "MATLAB reference workbooks, which are not redistributable, by tooling that "
          "is likewise kept local (README section 1). This project documents its "
          "verification rather than offering to reproduce it. If you hold the "
          "data, point COM_STAGE_ORACLES at it and this runs for real."
          % os.path.relpath(ORACLE, _ROOT))
    finish()

with io.open(ORACLE, encoding='utf-8') as _f:
    ORA = json.load(_f)

CASES = ORA['cases']
CMAP = ORA['column_map']

with io.open(REGISTRY, encoding='utf-8-sig') as _f:
    _REG_PRE = {x['name'] if isinstance(x, dict) else x
                for x in json.load(_f)['functions']}
_REG = _REG_PRE

check("oracle_covers_the_whole_reference_set",
      len(CASES) >= 208,
      "only %d oracle cases; the reference corpus has 208 and the oracle is "
      "meant to mirror it. Regenerate with "
      "the local oracle extractor" % len(CASES))

_unresolved = [c['sheet'] + ':' + str(c['row'])
               for c in CASES if not c['case_id']]
check("every_oracle_case_maps_to_a_known_case_id",
      not _unresolved,
      "%d oracle case(s) could not be tied back to report_data/compare.csv: "
      "%s -- an unmapped case cannot be provenance-checked or run live"
      % (len(_unresolved), _unresolved[:5]))

_by_cond = _collections.Counter(c['cond'] for c in CASES)
check("oracle_balanced_across_crosstalk_conditions",
      set(_by_cond) == {'wXtalk', 'woXtalk'} and min(_by_cond.values()) >= 100,
      "crosstalk coverage is %s -- crosstalk-only defects (the 8 sampling-"
      "phase divergences are all wXtalk) need both sides represented"
      % dict(_by_cond))

_by_test = _collections.Counter(c['test'] for c in CASES)
check("oracle_covers_all_four_package_cases",
      len(_by_test) >= 4 and min(_by_test.values()) >= 40,
      "package-case coverage is %s -- the FOM discrepancy tracked the package, "
      "so a gap here hides exactly the class of defect that was hardest to "
      "find" % dict(_by_test))

VMAP = ORA.get('vector_map', {})
VEC_STAGES = {v['stage'] for v in VMAP.values()}

check("oracle_has_vector_families",
      len(VMAP) >= 14,
      "only %d vector families; the tap sets and package vectors are the "
      "quantities a scalar cannot anchor. Regenerate with "
      "the local oracle extractor" % len(VMAP))

_missing_vec = sorted({v['function'] for v in VMAP.values()} - _REG_PRE)
check("oracle_vector_functions_all_exist",
      not _missing_vec,
      "vector families attributed to functions not in the registry: %s"
      % _missing_vec)

# Every family must be present in every case, at a consistent length.
_EXPECTED_LEN = {
    'RxFFE': 87, 'floating_tap_locations': 8, 'pkg_Z_c': 8,
    'CTLE_zero_poles': 3, 'C_diepad': 6, 'L_comp': 6,
    'Pkg_len_TX': 4, 'Pkg_len_RX': 4, 'Pkg_len_NEXT': 4, 'Pkg_len_FEXT': 4,
    'C_bump': 2, 'C_v': 2, 'R_diepad': 2,
}
_len_bad, _absent, _holed = [], [], []
for c in CASES:
    vecs = c.get('vectors', {})
    for fam, want in _EXPECTED_LEN.items():
        if fam not in vecs:
            _absent.append('%s/%s' % (c['case_id'], fam))
        elif len(vecs[fam]) != want:
            _len_bad.append('%s/%s len %d != %d'
                            % (c['case_id'], fam, len(vecs[fam]), want))
    for fam, vec in vecs.items():
        if any(v is None for v in vec):
            _holed.append('%s/%s' % (c['case_id'], fam))

check("vector_families_present_in_every_case",
      not _absent,
      "%d missing family/case combinations, e.g. %s -- a family that silently "
      "disappears (C_diepad and L_comp are stored as TEXT and are dropped by a "
      "naive numeric check) leaves that stage unanchored"
      % (len(_absent), _absent[:5]))

check("vector_lengths_are_as_expected",
      not _len_bad,
      "%d vectors have an unexpected length, e.g. %s -- a shortened tap vector "
      "is exactly engine defect #3, where the RxFFE floating-tap array was "
      "sized by tap COUNT (23) instead of SPAN (87)"
      % (len(_len_bad), _len_bad[:5]))

check("vectors_have_no_interior_holes",
      not _holed,
      "%d vectors contain an interior blank, e.g. %s -- a hole means the "
      "family was read misaligned, and every tap after it is shifted"
      % (len(_holed), _holed[:5]))

# The two workbooks record Tx FFE at different granularity. Pin it, so a future
# MATLAB export that changes the convention is reported rather than silently
# reinterpreted.
_tx_by_cond = _collections.defaultdict(set)
for c in CASES:
    v = c.get('vectors', {}).get('TXLE_taps')
    if v:
        _tx_by_cond[c['cond']].add(len(v))
check("txffe_granularity_matches_the_known_workbook_convention",
      _tx_by_cond.get('wXtalk') == {4} and _tx_by_cond.get('woXtalk') == {1},
      "Tx FFE tap-vector lengths are %s; expected wXtalk={4} (TXLE_taps_1..4) "
      "and woXtalk={1} (a single TXLE_taps column). If this changed, the two "
      "workbooks no longer record the same quantity the same way and the tap "
      "oracle needs re-reading." % {k: sorted(v) for k, v in _tx_by_cond.items()})

_stages = {v['stage'] for v in CMAP.values()} | VEC_STAGES
check("oracle_spans_the_pipeline",
      len(_stages) >= 6,
      "oracle covers only %d stages (%s) -- a stage with no MATLAB anchor is a "
      "stage where a regression is invisible" % (len(_stages), sorted(_stages)))

_unknown = sorted({v['function'] for v in CMAP.values()} - _REG)
check("oracle_functions_all_exist",
      not _unknown,
      "the oracle attributes values to functions that are not in the registry: "
      "%s -- either a MATLAB update removed them, or the mapping is wrong"
      % _unknown)

_thin = sorted(c['case_id'] or c['sheet'] + ':' + str(c['row'])
               for c in CASES if len(c['values']) < 20)
check("every_oracle_case_is_populated",
      not _thin,
      "oracle cases carrying fewer than 20 values: %s -- a sparse case gives "
      "false confidence" % _thin[:6])

# --------------------------------------------------------------- layer 2
if not os.path.exists(COMPARE):
    # Same reasoning as the oracle: compare.csv carries the MATLAB com_mat /
    # fom_mat columns, so it is not in the repository either. Without it the
    # provenance cross-check simply does not run.
    print("   provenance cross-check skipped: report_data/compare.csv not present")
else:
    with io.open(COMPARE, encoding='utf-8') as _f:
        _CMP = {r['case_id']: r for r in csv.DictReader(_f)}

    _matched = _mismatched = 0
    _bad = []
    # oracle column -> the compare.csv column holding the SAME MATLAB value
    _PAIRS = [('COM_dB', 'com_mat', 1e-9), ('FOM', 'fom_mat', 1e-9),
              ('itick', 'itick_mat', 0.0)]
    for c in CASES:
        row = _CMP.get(c['case_id'] or '')
        if not row:
            continue
        for okey, ckey, tol in _PAIRS:
            if okey not in c['values'] or not row.get(ckey):
                continue
            a, b = float(c['values'][okey]), float(row[ckey])
            if abs(a - b) <= tol:
                _matched += 1
            else:
                _mismatched += 1
                _bad.append('%s %s: oracle %.10g vs compare.csv %.10g'
                            % (c['case_id'], okey, a, b))

    check("oracle_values_match_the_comparison_table",
          _mismatched == 0 and _matched > 0,
          "%d of %d MATLAB values disagree with report_data/compare.csv: %s. "
          "The workbooks repeat header names (COM_dB, DER_thresh, rtmin, "
          "DER_MLSE); reading the wrong duplicate is exactly how fake 2.58 dB "
          "discrepancies appeared during the correlation."
          % (_mismatched, _matched + _mismatched, _bad[:4]))

    print("   provenance: %d MATLAB values cross-checked against compare.csv"
          % _matched)

# --------------------------------------------------------------- layer 3
_live = os.environ.get('COM_ORACLE_LIVE') == '1'
if not _live:
    print("\nlive stage comparison SKIPPED (set COM_ORACLE_LIVE=1 to enable).")
    print("It needs the gitignored channel set and runs the engine per case;")
    print("the standing full-corpus equivalent is:")
    print("   run the local comparison harness to build it")
else:
    import sicopr  # noqa: F401  (imported only when actually running the engine)

    _manifest = os.path.join(_ROOT, 'matlab_compare_results', 'manifest.json')
    if not os.path.exists(_manifest):
        check("live_run_inputs_available", False,
              "COM_ORACLE_LIVE=1 but matlab_compare_results/manifest.json is "
              "absent; build it with the local comparison harness")
    else:
        with io.open(_manifest, encoding='utf-8') as _f:
            _MAN = {c['case_id']: c for c in json.load(_f)['cases']}
        _ran = 0
        for c in CASES:
            job = _MAN.get(c['case_id'] or '')
            if not job or not os.path.exists(job['thru']):
                continue
            files = [job['thru']] + list(job['fext']) + list(job['next'])
            res = sicopr._run_com(job['config'], len(job['fext']),
                               len(job['next']), files, export_mat=False)
            r = res[0] if isinstance(res, (list, tuple)) else res
            _ran += 1
            for okey, attr, tol in (('COM_dB', 'COM', 2e-1),
                                    ('VEO_mV', 'VEO', 1e-1),
                                    ('VEC_dB', 'VEC', 1e-1)):
                if okey not in c['values'] or not hasattr(r, attr):
                    continue
                got = float(getattr(r, attr))
                want = float(c['values'][okey])
                check("live__%s__%s" % (c['case_id'], okey),
                      abs(got - want) <= tol,
                      "Python %.6f vs MATLAB %.6f (tolerance %g)"
                      % (got, want, tol))
        check("live_run_covered_some_cases", _ran > 0,
              "no oracle case had its channel files present on disk")

print("\n%d oracle cases, %d MATLAB quantities each, %d stages"
      % (len(CASES), len(CMAP), len(_stages)))
finish()
