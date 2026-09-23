"""Triage aid: which surviving `.copy()` mutants are real gaps?

109 of the 116 `drop_dot_copy` mutants survived. That number is an UPPER BOUND
on the real gap, because some of those copies are defensive on a value nothing
ever writes, and removing them genuinely changes nothing. Those are equivalent
mutants, not sleeping tests.

Arguing 109 sites by hand is how a list becomes a chore and then a lie, so the
split is computed. For each surviving site this asks what the copy is FOR:

  load-bearing   the copied-to name is written in place afterwards (subscript
                 assignment, augmented assignment, an in-place method, `out=`).
                 Drop the copy and that write lands on the SOURCE, which is the
                 caller's array. This is the aliasing defect that accounted for
                 5 of the 8 defects found in 2026-08. A surviving mutant here is
                 a REAL GAP and wants a test.

  escaping       the name is returned, or stored into a struct attribute, and
                 not written locally. The callee is clean but the CALLER may
                 write it, so safety depends on code this analysis cannot see.
                 Needs a human, and is reported separately rather than being
                 quietly filed under either heading.

  defensive      the name is neither written in place nor escapes. Dropping the
                 copy cannot change an observable result, so the mutant is
                 equivalent and the test was right not to fail.

This is a TRIAGE AID and not a verification method: it cannot be given a
negative control, so it closes no rows. It says where to look. What closes a
row is a test that fails when the copy is removed.

    python com_functions/verification/triage_copies.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import ast
import collections
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
FN = os.path.join(_ROOT, 'com_functions', 'fn')
LAST_RUN = os.path.join(_HERE, '_last_run.json')

# numpy methods that write the array in place. A copy feeding one of these is
# load-bearing by definition.
INPLACE_METHOD = {'sort', 'fill', 'resize', 'put', 'itemset', 'setfield',
                  'partition', 'byteswap'}


def enclosing_function(tree, lineno):
    """The innermost FunctionDef containing this line."""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, 'end_lineno', node.lineno)
        if node.lineno <= lineno <= end:
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def copy_target(fnode, lineno):
    """The name assigned from a `.copy()` on this line, if it is a simple one."""
    for node in ast.walk(fnode):
        if not isinstance(node, ast.Assign) or node.lineno != lineno:
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr == 'copy':
                return node.targets[0].id
    return None


def written_in_place(fnode, name, after):
    """Is `name` written in place after line `after`?"""
    for node in ast.walk(fnode):
        if getattr(node, 'lineno', 0) <= after:
            continue
        # name[...] = ...   and   name[...] += ...
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for t in targets:
            base = t
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id == name and base is not t:
                return True
            # name += ... rebinds for immutables but mutates a numpy array
            if isinstance(node, ast.AugAssign) and isinstance(t, ast.Name) \
                    and t.id == name:
                return True
        # name.sort(), np.place(name, ...), np.copyto(name, ...)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in INPLACE_METHOD \
                    and isinstance(f.value, ast.Name) and f.value.id == name:
                return True
            if isinstance(f, ast.Attribute) and f.attr in ('place', 'copyto',
                                                           'put', 'putmask'):
                if node.args and isinstance(node.args[0], ast.Name) \
                        and node.args[0].id == name:
                    return True
            for kw in node.keywords:
                if kw.arg == 'out' and isinstance(kw.value, ast.Name) \
                        and kw.value.id == name:
                    return True
    return False


def escapes(fnode, name, after):
    """Is `name` returned or stored into a struct attribute?"""
    for node in ast.walk(fnode):
        if getattr(node, 'lineno', 0) < after:
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Name) and sub.id == name:
                    return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, (ast.Attribute, ast.Subscript)):
                    for sub in ast.walk(node.value):
                        if isinstance(sub, ast.Name) and sub.id == name:
                            return True
    return False


def main():
    if not os.path.isfile(LAST_RUN):
        raise SystemExit(
            'no _last_run.json. Run:\n'
            '  python com_functions/verification/mutations.py --run '
            '--json com_functions/verification/_last_run.json')
    rows = json.load(io.open(LAST_RUN, encoding='utf-8'))
    sites = [r for r in rows
             if r['op'] == 'drop_dot_copy' and r['outcome'] == 'survived']

    buckets = collections.defaultdict(list)
    for r in sites:
        impl = os.path.join(FN, r['fn'], 'py_impl.py')
        src = io.open(impl, encoding='utf-8').read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            buckets['unparsed'].append((r['fn'], r['line'], '?'))
            continue
        fnode = enclosing_function(tree, r['line'])
        if fnode is None:
            buckets['module-level'].append((r['fn'], r['line'], '?'))
            continue
        name = copy_target(fnode, r['line'])
        if name is None:
            # e.g. foo(a.copy()) -- the copy is consumed by the callee, so
            # whether it matters depends on the callee, not on this function.
            buckets['passed-to-a-call'].append((r['fn'], r['line'], '-'))
            continue
        if written_in_place(fnode, name, r['line']):
            buckets['load-bearing'].append((r['fn'], r['line'], name))
        elif escapes(fnode, name, r['line']):
            buckets['escaping'].append((r['fn'], r['line'], name))
        else:
            buckets['defensive'].append((r['fn'], r['line'], name))

    print('%d surviving drop_dot_copy mutants\n' % len(sites))
    order = ['load-bearing', 'escaping', 'passed-to-a-call', 'defensive',
             'module-level', 'unparsed']
    for k in order:
        if buckets[k]:
            print('  %-18s %3d' % (k, len(buckets[k])))
    print()

    for k in order:
        if not buckets[k]:
            continue
        print('\n=== %s (%d) ===' % (k, len(buckets[k])))
        for fn, line, name in sorted(buckets[k]):
            print('   %-36s line %-5d %s' % (fn, line, name))

    print('\nload-bearing sites are REAL GAPS: a test there should fail when '
          'the copy is removed, and none does.')
    print('defensive sites are candidate equivalent mutants for '
          'equivalent_mutants.md, each still needing its one-sentence '
          'argument.')


if __name__ == '__main__':
    sys.exit(main())
