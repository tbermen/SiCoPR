"""A behaviour change to a py_impl.py must come with a change to its test.

The failure this exists for, observed 2026-09-22: four agents were working in
parallel and one was cut off mid-function by a rate limit. It left
OptFom_Calc_Noise/py_impl.py with two real behaviour changes and no test for
either. Everything still passed, because the suite only runs the tests that
exist. An untested engine change is the single most dangerous state this
repository can be left in, and it is exactly the state an interruption
produces.

Noticing it depended on a human reading `git status` carefully. That is not a
control, so this is.

WHAT IT CHECKS
    For every com_functions/fn/<name>/py_impl.py modified in the working tree,
    com_functions/fn/<name>/test_verify.py must be modified too.

WHY AST EQUALITY RATHER THAN A TEXT DIFF
    Comment, docstring and whitespace edits are constant here -- most of this
    port's value is in the comments explaining what MATLAB does -- and
    demanding a test change for a reworded comment would be friction that
    teaches people to bypass the check. So the rule is BEHAVIOUR changed, and
    the proxy for behaviour is the parsed AST with docstrings stripped. A
    comment-only edit is invisible to it; a one-character change to an
    expression is not.

SCOPE
    The working tree only. Committed history is not re-litigated: this is a
    gate on what is about to be committed, which is the moment the state is
    still cheap to fix.

Run: python tests/test_change_discipline.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import ast
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_check import check, finish  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    p = subprocess.run(('git',) + args, cwd=ROOT, capture_output=True,
                       text=True, errors='replace')
    return p.stdout if p.returncode == 0 else None


def _strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            if (n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)
                    and isinstance(n.body[0].value.value, str)):
                n.body.pop(0)
    return tree


def _shape(src):
    """A comment- and docstring-free rendering, or None if it will not parse."""
    try:
        return ast.dump(_strip_docstrings(ast.parse(src)))
    except SyntaxError:
        return None


status = _git('status', '--porcelain')

if status is None:
    check('change_discipline_needs_git', True,
          'not a git checkout; nothing to compare against')
else:
    changed = set()
    for line in status.split('\n'):
        path = line[3:].strip().replace('\\', '/')
        if path.startswith('com_functions/fn/') and path.endswith('.py'):
            parts = path.split('/')
            if len(parts) == 4:
                changed.add((parts[2], parts[3]))

    impl_changed = {fn for fn, f in changed if f == 'py_impl.py'}
    test_changed = {fn for fn, f in changed if f == 'test_verify.py'}

    # Only the ones whose BEHAVIOUR moved. A reworded comment is not a
    # behaviour change and must not demand a test edit.
    behaviour_changed = set()
    unparsable = []
    for fn in sorted(impl_changed):
        rel = 'com_functions/fn/%s/py_impl.py' % fn
        head = _git('show', 'HEAD:' + rel)
        now = io.open(os.path.join(ROOT, rel), encoding='utf-8').read()
        if head is None:                       # newly added file
            behaviour_changed.add(fn)
            continue
        a, b = _shape(head), _shape(now)
        if b is None:
            unparsable.append(fn)
            continue
        if a != b:
            behaviour_changed.add(fn)

    check('every_changed_py_impl_parses',
          not unparsable,
          'these py_impl.py files do not parse, so nothing downstream of them '
          'can be trusted: %s' % sorted(unparsable))

    naked = sorted(behaviour_changed - test_changed)
    check('no_behaviour_change_without_a_test',
          not naked,
          'these have a BEHAVIOUR change in py_impl.py and no change to their '
          'test_verify.py: %s. An untested engine change is how an interrupted '
          'session leaves the tree, and the suite cannot see it because it only '
          'runs the tests that exist. Add the test, or if the change really is '
          'covered by an existing assertion, say so in the commit message and '
          'touch the test file with that note.' % naked)

    print('\nworking tree: %d py_impl changed, %d of them behaviourally, '
          '%d test files changed'
          % (len(impl_changed), len(behaviour_changed), len(test_changed)))

finish()
