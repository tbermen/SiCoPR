"""Layer 5b -- every MATLAB builtin the reference uses must be classified.

Row type (b) of the verification contract, and the link that would have caught
the `np.std` defect at its source.

`sigma = std(x)` and `sigma = np.std(x)` read alike and are not alike. No amount
of careful reading finds that, because reading is what produced it. What finds
it is being forced to write down, once, what MATLAB's `std` does and what the
port must therefore do -- and then never being allowed to forget.

## Why this is a gate and not a list

`tools/translation_coverage.py` carries a hand-maintained BUILTIN set. A hand
list is the reviewer's memory in a variable, and memory is exactly what failed
in 2026-07, when 147 of 159 functions were marked EQUIVALENT from readings and
an oracle pass later found real divergences in 27 of them.

So this does not maintain a list of "the ones known to differ". It parses the
reference, extracts EVERY builtin actually used, and fails on anything absent
from `com_functions/verification/builtins.md`. Adding a MATLAB version that
uses a new builtin fails the suite until someone classifies it. Enforced that
way it needs no adherence at all.

On the first run the hand list was missing 88 of the 180 names in use, two of
which had already caused real defects here: `eps` (MATLAB's bare `eps` is
`eps(1)` = 2.22e-16, not `tiny` = 2.2e-308, a factor of 1e292 at the DC point in
three inlined make_pkg copies) and `erfcinv` (accurate in MATLAB, INACCURATE in
Octave in the tail, so the oracle must not be trusted there).

## The consequence of a `differs` verdict

A construct classified `differs` carries its rule, and **every function whose
Python touches it closes only by an oracle-backed test**. A reading cannot close
it. That rule is enforced in the contract, not here; this gate's job is to make
sure the classification exists at all.

## Resolving MATLAB's call-or-index ambiguity

MATLAB spells indexing and calling the same way, so `x(3)` looks like a call.
A name is a builtin only after subtracting the reference's own functions,
keywords, assigned locals, loop variables and the function's PARAMETERS. The
parameter case is not optional: without it `chdata` looks like a builtin used
81 times.

Names that survive all that and are still unknown are usually one of two real
things, and both matter: a genuinely unclassified builtin, or a function the
reference CALLS BUT NEVER DEFINES (`WIENER_HOPF_MMSE`, `conv_fct_TEST`), which
is an upstream defect for the COM ad hoc.

    python tests/test_builtin_classification.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import collections
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from audit_check import check, finish       # noqa: E402
import translation_coverage as tc           # noqa: E402

CLASSIFICATION = os.path.join(_ROOT, 'com_functions', 'verification',
                              'builtins.md')

KEYWORD = set('if elseif else end for while switch case otherwise break '
              'continue return function global persistent try catch '
              'parfor spmd do until'.split())

CALL = re.compile(r'(?<![\w.])([A-Za-z_]\w*)\s*\(')
ASSIGN = re.compile(r'(?m)^[^%\n]*?([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*=(?!=)')
BRACKET = re.compile(r'\[([^\]]*)\]\s*=')
FORVAR = re.compile(r'(?m)\bfor\s+([A-Za-z_]\w*)\s*=')
HEADER = re.compile(r'function\s*(?:\[([^\]]*)\]|([A-Za-z_]\w*))?\s*=?\s*'
                    r'([A-Za-z_]\w*)\s*\(([^)]*)\)')
# `param.x` and `P(k).field` mean the name is data. Without these two, struct
# variables read as builtins and the gate cries wolf, which teaches everyone to
# ignore it.
STRUCTVAR = re.compile(r'([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*\.\s*[A-Za-z_]')
GLOBALS = re.compile(r'(?m)^\s*global\s+([^\n%;]*)')

# A verdict must be one of these. `differs` obliges an oracle-backed closure.
VERDICTS = {'same', 'differs', 'no-numeric-result', 'undefined-upstream',
            'local-variable'}


def versions():
    v = json.load(io.open(os.path.join(_ROOT, 'VERSION.json'),
                          encoding='utf-8'))
    return v['supported_matlab_versions'], v['references']


def strip_matlab(src):
    """Blank out comments and string literals.

    MATLAB's apostrophe is BOTH the transpose operator and the string
    delimiter, so a regex cannot pair quotes. `param.package_testcase_i'` is a
    transpose, and a regex that treats it as an opening quote swallows the rest
    of the line and mis-lexes everything after it. That is what made `P_a_n`,
    `TS_SRCH_MODE`, `defined` and `num_ui_RXFF_noise` look like builtins: each
    is really a name INSIDE a format string.

    The disambiguation is MATLAB's own: an apostrophe directly after a value --
    an identifier, a digit, a closing bracket, a dot or another transpose -- is
    transpose. Anywhere else it opens a string.
    """
    out = []
    i, n = 0, len(src)
    prev = ''
    while i < n:
        c = src[i]
        if c == '%':                       # comment to end of line
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
            continue
        if c == '\n':
            out.append(c)
            i += 1
            prev = ''                      # a transpose cannot open a line
            continue
        if src.startswith('...', i):
            # A continuation ends in '.', and '.' means the NEXT apostrophe is
            # a transpose. It is not: the next line resumes at an operand, so a
            # quote there opens a string. Missing this made every name inside a
            # continued format string read as a builtin.
            j = src.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
            prev = ''
            continue
        if c == '"':                       # double-quoted string
            j = i + 1
            while j < n and src[j] != '"':
                j += 1
            out.append(' ' * (min(j + 1, n) - i))
            i = j + 1
            prev = '"'
            continue
        if c == "'":
            if prev and (prev.isalnum() or prev in "_)]}.'"):
                out.append(c)              # transpose
                i += 1
                prev = "'"
                continue
            j = i + 1                      # string literal, '' escapes a quote
            while j < n:
                if src[j] == "'":
                    if j + 1 < n and src[j + 1] == "'":
                        j += 2
                        continue
                    break
                if src[j] == '\n':
                    break
                j += 1
            out.append(' ' * (min(j + 1, n) - i))
            i = j + 1
            prev = "'"
            continue
        out.append(c)
        if not c.isspace():
            prev = c
        i += 1
    return ''.join(out)


def builtins_used(path):
    """{name: use count} for every builtin the reference actually calls."""
    fns = tc.matlab_functions(path)
    used = collections.Counter()
    for _name, body in fns.items():
        b = strip_matlab(body)
        local = set(ASSIGN.findall(b)) | set(FORVAR.findall(b)) | KEYWORD
        for grp in BRACKET.findall(b):
            local |= set(re.findall(r'[A-Za-z_]\w*', grp))
        # A struct is a variable: `param.x` means param is data, not a call.
        # Without this, names indexed as `P(k).field` read as builtins.
        local |= set(STRUCTVAR.findall(b))
        for grp in GLOBALS.findall(b):
            local |= set(re.findall(r'[A-Za-z_]\w*', grp))
        h = HEADER.search(b)
        if h:
            for grp in (h.group(1), h.group(2), h.group(4)):
                if grp:
                    local |= set(re.findall(r'[A-Za-z_]\w*', grp))
        for nm in CALL.findall(b):
            if nm not in fns and nm not in local:
                used[nm] += 1
    return used, fns


def classified():
    """{name: verdict} parsed from the markdown table."""
    if not os.path.isfile(CLASSIFICATION):
        return {}
    out = {}
    for line in io.open(CLASSIFICATION, encoding='utf-8'):
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 2:
            continue
        name, verdict = cells[0].strip('`'), cells[1]
        if verdict in VERDICTS:
            out[name] = verdict
    return out


def main():
    supported, refs = versions()
    known = classified()

    check('the_classification_file_exists',
          bool(known),
          'com_functions/verification/builtins.md is missing or has no '
          'classified rows. Every MATLAB builtin the reference uses must be '
          'classified there; that file is what makes a semantic difference '
          'impossible to forget.')

    bad = sorted(n for n, v in known.items() if v not in VERDICTS)
    check('every_verdict_is_from_the_fixed_vocabulary',
          not bad, 'unrecognised verdicts: %s' % bad)

    all_unclassified = {}
    for ver in supported:
        path = os.path.join(_ROOT, refs[ver]['file'])
        if not os.path.isfile(path):
            print('  %s: reference not present, skipped' % ver)
            continue
        used, fns = builtins_used(path)
        missing = sorted(n for n in used if n not in known)
        print('\n%s: %d functions, %d distinct builtins used, %d classified, '
              '%d NOT classified' % (ver, len(fns), len(used),
                                     len(used) - len(missing), len(missing)))
        for n in missing:
            all_unclassified.setdefault(n, []).append('%s(%d)' % (ver, used[n]))

    if all_unclassified:
        print('\nunclassified:')
        for n in sorted(all_unclassified):
            print('    %-30s %s' % (n, ','.join(all_unclassified[n])))

    check('every_builtin_the_reference_uses_is_classified',
          not all_unclassified,
          'these names are called by the reference and do not appear in '
          'builtins.md. Classify each one as same / differs / '
          'no-numeric-result / undefined-upstream, with its rule. An '
          'unclassified builtin is how np.std happened: nobody had ever been '
          'made to write down what MATLAB std does. Names: %s'
          % sorted(all_unclassified))

    # A `differs` verdict is the load-bearing one, so it must carry a rule.
    thin = []
    for line in io.open(CLASSIFICATION, encoding='utf-8'):
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) >= 3 and cells[1] == 'differs' and len(cells[2]) < 20:
            thin.append(cells[0].strip('`'))
    check('every_differs_verdict_states_its_rule',
          not thin,
          'these are classified `differs` with no rule written down, which '
          'records the alarm and throws away the information: %s' % thin)

    finish()


if __name__ == '__main__':
    main()
