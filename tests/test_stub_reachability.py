"""Layer 4e -- is anything running a STUB while reading as though it is not?

Several translated functions take their heavy dependencies by injection:

    def COM_FD_to_TD(chdata, param, OP, _s21_to_impulse_DC_fn=None, ...):
        s21_fn = (_s21_to_impulse_DC_fn if _s21_to_impulse_DC_fn is not None
                  else _s21_to_impulse_DC)      # <- a STUB

That is a good pattern for unit testing, and a silent lie for any caller that
forgets to inject. The fallback is not a thin shim: COM_FD_to_TD's is a
"minimal s21->impulse via ifft" standing in for the whole of
s21_to_impulse_DC, and it feeds the PRIMARY pulse responses.

This gate asks two questions.

## 1. Does the engine inject every one? (the serious question)

`sicopr.py` wires each such function with a `partial(...)` naming the real
implementations. A dependency missing from that partial means the ENGINE
silently runs a stub, which would be a defect of the first order. Checked here
so it can never be introduced.

## 2. Which TEST call sites call bare? (the quiet one)

On 2026-09-23 `tests/conftest.py` called `sicopr.COM_FD_to_TD(chdata, param,
OP)` with no injection, so the Stage-4 checkpoints and everything chaining off
`reference_result` asserted on a stubbed pipeline while reading as though they
validated the engine. Fixed in 208ca72. Those tests also SKIP without
COM_TEST_FIXTURES, so the stub was invisible twice over.

A bare call is not automatically wrong: a unit test may mean to drive the stub,
and several do. So the set is PINNED rather than forbidden. A new bare call
fails by name, and one that starts injecting must be removed from the set,
which holds the gain.

    python tests/test_stub_reachability.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import ast
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from audit_check import check, finish   # noqa: E402

FN = os.path.join(_ROOT, 'com_functions', 'fn')

# `<local> = _<dep>_fn if _<dep>_fn is not None else <fallback>`
# Both docstring conventions are in use: "Stub: ..." in COM_FD_to_TD
# and "Stub for X ..." in get_PSDs. Matching only the first missed all
# three of get_PSDs' stubs and reported it as 0 stub-backed.
STUB_DOC = re.compile('stub', re.IGNORECASE)

INJECT = re.compile(
    r'(\w+)\s*=\s*(_\w+_fn)\s+if\s+\2\s+is not None\s+else\s+(\w+)')

# Test call sites that deliberately drive the stub. Each entry is
# "<function>:<file>:<line>". Remove an entry when the call starts injecting;
# never add one to silence a new bare call without saying why below.
#
# get_PSDs' five were found by this gate on 2026-09-23 and are real: its unit
# tests exercise the stubs, so any value they pin is stub output rather than
# the engine's. They are pinned here rather than fixed blind, because changing
# what those tests drive changes what they assert.
KNOWN_BARE = frozenset([
    'FOM_rxffe_floating_taps:com_functions/fn/FOM_rxffe_floating_taps/test_verify.py:1',
    'get_PSDs:com_functions/fn/get_PSDs/test_verify.py:127',
    'get_PSDs:com_functions/fn/get_PSDs/test_verify.py:151',
    'get_PSDs:com_functions/fn/get_PSDs/test_verify.py:173',
    'get_PSDs:com_functions/fn/get_PSDs/test_verify.py:220',
    'get_PSDs:com_functions/fn/get_PSDs/test_verify.py:246',
])


def stub_functions(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = (ast.get_docstring(node) or '').lstrip()
            # BOTH conventions are in use: "Stub: ..." in COM_FD_to_TD and
            # "Stub for X ..." in get_PSDs. Matching only the first missed all
            # three of get_PSDs' stubs and reported it as 0 stub-backed, which
            # is the sort of quiet under-count this gate exists to prevent.
            if STUB_DOC.match(doc):
                out.add(node.name)
    return out


def survey():
    """{fn dir: (injectable params, stub-backed params)}"""
    out = {}
    for d in sorted(os.listdir(FN)):
        p = os.path.join(FN, d, 'py_impl.py')
        if not os.path.isfile(p):
            continue
        src = io.open(p, encoding='utf-8', errors='replace').read()
        try:
            stubs = stub_functions(ast.parse(src))
        except SyntaxError:
            continue
        deps = {m.group(2): m.group(3) for m in INJECT.finditer(src)}
        if deps:
            out[d] = (deps, {k for k, v in deps.items() if v in stubs})
    return out


def call_sites(fn_name, path):
    """Line numbers where `fn_name(` is called without any `_fn=` keyword."""
    if not os.path.isfile(path):
        return []
    t = io.open(path, encoding='utf-8', errors='replace').read()
    hits = []
    for m in re.finditer(r'(?<![\w.])%s\(' % re.escape(fn_name), t):
        seg = t[m.start():m.start() + 4000]
        depth, end = 0, None
        for i, c in enumerate(seg):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        call = seg[:end] if end is not None else seg
        if '_fn=' not in call:
            hits.append(t[:m.start()].count('\n') + 1)
    return hits


found = survey()
engine = io.open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8',
                 errors='replace').read()

check('injectable_dependencies_were_found', bool(found),
      'no function with an injectable dependency was detected, so every check '
      'below is vacuous. The INJECT pattern probably stopped matching.')

# ---- 1. the engine must inject every one -----------------------------------
engine_gaps = []
for d, (deps, stubbed) in sorted(found.items()):
    m = re.search(r'_wired_%s\s*=\s*partial\(\s*%s\s*,(.*?)\n\n'
                  % (re.escape(d), re.escape(d)), engine, re.S)
    if not m:
        continue                      # not wired through a partial; nothing to check
    wired = set(re.findall(r'(_\w+_fn)\s*=', m.group(1)))
    missing = sorted(set(deps) - wired)
    if missing:
        engine_gaps.append('%s: the engine does not inject %s'
                           % (d, ', '.join(missing)))

check('the_engine_injects_every_dependency',
      not engine_gaps,
      'these dependencies fall back to their STUB on the engine path, so a '
      'real run would compute with a placeholder. sicopr.py builds each '
      '_wired_<name> partial; add the missing keyword there: %s' % engine_gaps)

# ---- 2. pinned set of bare test call sites ---------------------------------
bare = set()
for d in found:
    for rel in ('com_functions/fn/%s/test_verify.py' % d, 'tests/conftest.py'):
        for line in call_sites(d, os.path.join(_ROOT, rel)):
            bare.add('%s:%s:%d' % (d, rel, line))

new = sorted(bare - KNOWN_BARE)
check('no_new_bare_call_to_a_stub_backed_function',
      not new,
      'these call a function with stub-backed dependencies and inject none of '
      'them, so they run the STUB. If that is deliberate, add the entry to '
      'KNOWN_BARE with the reason; if not, inject the real functions the way '
      'sicopr.py does: %s' % new)

fixed = sorted(KNOWN_BARE - bare)
check('known_bare_call_list_is_current',
      not fixed,
      'these now inject their dependencies, or have moved. Delete them from '
      'KNOWN_BARE so the gain is held: %s' % fixed)

print('\n%d function(s) take injectable dependencies; %d bare test call '
      'site(s), %d pinned.' % (len(found), len(bare), len(KNOWN_BARE)))
for d, (deps, stubbed) in sorted(found.items()):
    print('   %-28s %d injectable, %d stub-backed' % (d, len(deps), len(stubbed)))

finish()
