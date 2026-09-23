"""A function's test must exercise the options its implementation branches on.

`interp_Sparam` has a unit test. The test passes phase methods `old` and
`interp_to_DC`; the engine's configured default is
`extrap_cubic_to_dc_linear_to_inf`, and that branch holds all three `std`
calls. So the function was tested, the default path was not, and the N-vs-N-1
defect sat in code no unit test ever executed. The 208-case corpus did not
reach it either, because every channel in that corpus starts at DC.

Integration coverage is the wrong tool for this: it only exercises the options
the corpus happens to select, and a corpus is a sample of the world, not a
cover of the code. Option branches are enumerable from the source, so they can
be covered deliberately at unit level.

This gate lists, per translated function, the option strings its `py_impl.py`
compares against and whether its own `test_verify.py` ever passes one. A
function whose default option is untested is the shape the `std` defect had.

The set of uncovered options is pinned, not just their number, so a failure
names the branch that changed. Giving one a test means deleting its entry;
adding a branch without a test fails here. Coverage can only improve.

The two options every shipped workbook selects are held to a stricter rule:
they must always be tested. That one is a `check`, not a ledger entry.

Run: python tests/test_option_coverage.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from audit_check import check, finish   # noqa: E402

FN = os.path.join(_ROOT, 'com_functions', 'fn')

# Strings a py_impl compares against: == 'x', != 'x', in ('x', ...).
OPT = re.compile(r"(?:==|!=)\s*'([a-zA-Z_][a-zA-Z0-9_]{3,})'"
                 r"|in\s*\(\s*'([a-zA-Z_][a-zA-Z0-9_]{3,})'")

# Not options: module guards, literals that happen to be compared.
NOISE = {'__main__', 'true', 'false', 'none'}

# Options the shipped configuration workbooks actually select. An untested
# option here is not a gap in the abstract: it is the path every run takes.
DEFAULTS = {
    'extrap_cubic_to_dc_linear_to_inf',   # OP.interp_sparam_phase
    'linear_trend_to_DC',                 # OP.interp_sparam_mag
}

# Pinned 2026-09-22, the state this gate was written against. Remove an entry
# when its option gains a test; never add one to make a red run go green.
# Pinning the set rather than a count lets the failure name the new branch.
KNOWN_UNCOVERED = frozenset([
])


def scan():
    rows = []
    for name in sorted(os.listdir(FN)):
        impl = os.path.join(FN, name, 'py_impl.py')
        test = os.path.join(FN, name, 'test_verify.py')
        if not os.path.isfile(impl):
            continue
        src = open(impl, encoding='utf-8', errors='replace').read()
        opts = {m.group(1) or m.group(2) for m in OPT.finditer(src)}
        opts = {o for o in opts if not o.isupper() and o.lower() not in NOISE}
        if not opts:
            continue
        tsrc = (open(test, encoding='utf-8', errors='replace').read()
                if os.path.isfile(test) else '')
        missing = sorted(o for o in opts if ("'%s'" % o) not in tsrc)
        rows.append((name, sorted(opts), missing))
    return rows


rows = scan()
uncovered = sorted((fn, o) for fn, _, missing in rows for o in missing)
n_opts = sum(len(o) for _, o, _ in rows)

print('%d functions branch on an option string, %d options in all, %d uncovered\n'
      % (len(rows), n_opts, len(uncovered)))
for fn, opts, missing in sorted(rows, key=lambda r: -len(r[2])):
    if missing:
        print('  %-34s %d of %d uncovered: %s'
              % (fn, len(missing), len(opts), ', '.join(missing[:4])))
print()

keys = {'%s:%s' % (fn, o) for fn, o in uncovered}
new = sorted(keys - KNOWN_UNCOVERED)
closed = sorted(KNOWN_UNCOVERED - keys)

check('no_new_untested_option_branch',
      not new,
      'these option branches are new and no test passes them: %s' % new)

check('known_uncovered_list_is_current',
      not closed,
      'these now have a test -- delete them from KNOWN_UNCOVERED so the gain '
      'is held: %s' % closed)

default_gaps = sorted({(fn, o) for fn, o in uncovered if o in DEFAULTS})
check('configured_default_options_are_tested',
      not default_gaps,
      "the option every shipped workbook selects is not passed by the "
      "function's own test: %s. This is the shape the np.std defect had -- "
      'interp_Sparam was tested, but not on the branch it actually runs, '
      'which is where all three std calls live. Closed 2026-09-22 by the '
      'default-path tests in interp_Sparam and s21_to_impulse_DC, both pinned '
      'to COM Octave; it must not reopen.'
      % ['%s: %s' % (f, o) for f, o in default_gaps])

finish()
