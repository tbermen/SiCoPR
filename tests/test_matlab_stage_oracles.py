"""MATLAB-anchored stage oracles: real reference values, no ask on anyone.

Every one of the 156 per-function test files asserts what Python already does.
None pins a MATLAB-produced number. That is the structural reason the eight
engine defects survived 876 unit tests: the suite could not tell "correct" from
"consistently wrong".

The ideal fix -- a per-function dump from the MATLAB side -- is a large favour
to ask of someone who gets nothing back from it. This gets most of the way
there from data already committed: the two reference workbooks under
tests/2_Results_COM_Matlab/ carry ~260 columns of genuine MATLAB output per
case, and many are the output of one identifiable stage.

tools/extract_matlab_oracles.py distils those into
tests/oracles/matlab_stage_oracles.json -- all 208 reference cases, 35 MATLAB
quantities each across 7 stages, ~300 KB, committed. This file checks it.

Three quantities are deliberately sparse and are NOT a extraction fault:
  TXLE_taps  104/208  a tap VECTOR, stored as text in one of the two workbooks
                      (the same two workbooks disagree on the Tx FFE tap set
                      while naming the same config_file -- an open question)
  sgm_rjit   160/208  blank where the case carries no random-jitter term
  sgm_xt     203/208  blank where the case has no crosstalk contribution

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

ORACLE = os.path.join(_HERE, 'oracles', 'matlab_stage_oracles.json')
COMPARE = os.path.join(_ROOT, 'report_data', 'compare.csv')
REGISTRY = os.path.join(_ROOT, 'com_functions', 'registry.json')

# --------------------------------------------------------------- layer 1
check("oracle_file_present",
      os.path.exists(ORACLE),
      "%s is missing -- regenerate with "
      "`python tools/extract_matlab_oracles.py`" % os.path.relpath(ORACLE, _ROOT))

if not os.path.exists(ORACLE):
    finish()

with io.open(ORACLE, encoding='utf-8') as _f:
    ORA = json.load(_f)

CASES = ORA['cases']
CMAP = ORA['column_map']

check("oracle_covers_the_whole_reference_set",
      len(CASES) >= 208,
      "only %d oracle cases; the reference corpus has 208 and the oracle is "
      "meant to mirror it. Regenerate with "
      "`python tools/extract_matlab_oracles.py`" % len(CASES))

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

_stages = {v['stage'] for v in CMAP.values()}
check("oracle_spans_the_pipeline",
      len(_stages) >= 6,
      "oracle covers only %d stages (%s) -- a stage with no MATLAB anchor is a "
      "stage where a regression is invisible" % (len(_stages), sorted(_stages)))

with io.open(REGISTRY, encoding='utf-8-sig') as _f:
    _REG = {x['name'] if isinstance(x, dict) else x
            for x in json.load(_f)['functions']}

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
    check("provenance_table_present", False,
          "report_data/compare.csv missing; cannot verify the oracle was "
          "extracted from the right workbook columns")
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
    print("   python tools/matlab_compare.py --validate --run --jobs 5")
else:
    import com  # noqa: F401  (imported only when actually running the engine)

    _manifest = os.path.join(_ROOT, 'matlab_compare_results', 'manifest.json')
    if not os.path.exists(_manifest):
        check("live_run_inputs_available", False,
              "COM_ORACLE_LIVE=1 but matlab_compare_results/manifest.json is "
              "absent; build it with `python tools/matlab_compare.py --validate`")
    else:
        with io.open(_manifest, encoding='utf-8') as _f:
            _MAN = {c['case_id']: c for c in json.load(_f)['cases']}
        _ran = 0
        for c in CASES:
            job = _MAN.get(c['case_id'] or '')
            if not job or not os.path.exists(job['thru']):
                continue
            files = [job['thru']] + list(job['fext']) + list(job['next'])
            res = com._run_com(job['config'], len(job['fext']),
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
