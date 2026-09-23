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
BASELINE_WEAK_COMPOSITES = 2
check('composite_gap_does_not_grow',
      len(weak_comp) <= BASELINE_WEAK_COMPOSITES,
      'composite functions without a value check rose from %d to %d: %s'
      % (BASELINE_WEAK_COMPOSITES, len(weak_comp), weak_comp))

check('composite_baseline_is_current',
      len(weak_comp) >= BASELINE_WEAK_COMPOSITES,
      'composite gap fell from %d to %d -- lower BASELINE_WEAK_COMPOSITES to '
      '%d so the gain is held. Remaining: %s'
      % (BASELINE_WEAK_COMPOSITES, len(weak_comp), len(weak_comp), weak_comp))

# Shape is its own axis. A test that pins every number still need not say how
# many there are, nor which orientation they come back in -- and MATLAB
# distinguishes a row from a column where a 1-D numpy array does not. Ratcheted
# rather than gated outright, because the leaf gap has to close first.
BASELINE_SHAPE_CHECKED = 123
n_shape = sum(1 for r in bearing if r['shape_checks'])
no_shape = sorted(r['function'] for r in bearing if not r['shape_checks'])

check('shape_coverage_does_not_fall',
      n_shape >= BASELINE_SHAPE_CHECKED,
      'functions whose tests check a shape fell from %d to %d. Still unchecked: '
      '%s' % (BASELINE_SHAPE_CHECKED, n_shape, no_shape))

check('shape_baseline_is_current',
      n_shape <= BASELINE_SHAPE_CHECKED,
      'shape coverage rose from %d to %d -- raise BASELINE_SHAPE_CHECKED to %d '
      'so the gain is held' % (BASELINE_SHAPE_CHECKED, n_shape, n_shape))

# The number that actually catches a library default: tests whose expected
# values came from EXECUTING the reference under Octave, not from reading the
# MATLAB. A test written from a reading cannot see that `std(x)` and
# `np.std(x)` differ, which is how the N-vs-N-1 defect survived to 2026-09-21.
# 2026-09-22 took this from 22 to 56 in one pass; ratcheted so it cannot slip.
BASELINE_ORACLE_BACKED = 134
n_oracle = sum(1 for r in bearing if r['grade'] == 'oracle')

check('oracle_coverage_does_not_fall',
      n_oracle >= BASELINE_ORACLE_BACKED,
      'functions checked against the executed reference fell from %d to %d'
      % (BASELINE_ORACLE_BACKED, n_oracle))

check('oracle_baseline_is_current',
      n_oracle <= BASELINE_ORACLE_BACKED,
      'oracle coverage rose from %d to %d -- raise BASELINE_ORACLE_BACKED to '
      '%d so the gain is held' % (BASELINE_ORACLE_BACKED, n_oracle, n_oracle))

finish()
