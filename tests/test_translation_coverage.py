"""Every leaf function in the MATLAB reference must have its values checked.

A leaf calls no other translated function, so it can be driven directly with
made-up inputs. These are the syntax translations -- the places where a MATLAB
builtin became a numpy call and a differing default can hide, which is exactly
what `std` becoming `np.std` was. There is no setup cost to hide behind, so a
leaf whose test only checks a shape is a gap with no excuse.

Composite functions are not gated here. They need their callees stood up, a
failure may belong to one of those, and the honest move is to fix the leaves
first. Their count is reported so it cannot drift unnoticed.

The grading lives in tools/translation_coverage.py; run that for the table.

Run: python tests/test_translation_coverage.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from audit_check import check, finish          # noqa: E402
import translation_coverage as tc              # noqa: E402

rows = tc.survey()
for r in rows:
    r['grade'] = tc.grade(r)

bearing = [r for r in rows if r['grade'] != 'no numeric result']
leaves = [r for r in bearing if r['kind'] == 'leaf']
weak_leaves = [r['function'] for r in leaves
               if r['grade'] in ('shape only', 'no test', 'not translated')]
composites = [r for r in bearing if r['kind'] == 'composite']
weak_comp = [r['function'] for r in composites
             if r['grade'] in ('shape only', 'no test', 'not translated')]

print('%d functions in the reference, %d translated, %d value-bearing'
      % (len(rows), sum(1 for r in rows if r['translated']), len(bearing)))
print('leaf %d, composite %d; %d checked against the executed reference'
      % (len(leaves), len(composites),
         sum(1 for r in bearing if r['grade'] == 'oracle')))

check('every_reference_function_is_translated',
      all(r['translated'] for r in rows),
      'not translated: %s' % [r['function'] for r in rows if not r['translated']])

check('every_leaf_function_checks_its_values',
      not weak_leaves,
      'these call nothing else, so they can be driven directly and their tests '
      'must pin a value rather than a shape: %s' % weak_leaves)

# Composites are reported, not gated. The number is pinned so that adding one
# without a value check is visible, and lowering it is the way to improve.
BASELINE_WEAK_COMPOSITES = 4
check('composite_gap_does_not_grow',
      len(weak_comp) <= BASELINE_WEAK_COMPOSITES,
      'composite functions without a value check rose from %d to %d: %s'
      % (BASELINE_WEAK_COMPOSITES, len(weak_comp), weak_comp))

check('composite_baseline_is_current',
      len(weak_comp) >= BASELINE_WEAK_COMPOSITES,
      'composite gap fell from %d to %d -- lower BASELINE_WEAK_COMPOSITES to '
      '%d so the gain is held. Remaining: %s'
      % (BASELINE_WEAK_COMPOSITES, len(weak_comp), len(weak_comp), weak_comp))

finish()
