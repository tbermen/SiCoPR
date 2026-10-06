"""The assembled engine must not lose a name its sources had.

`assemble_sicopr.py` concatenates every `py_impl.py` into one module. Two things
go wrong in that step that no per-function test can see, because each of those
tests imports its own `py_impl.py`, where the names are fine:

  * IMPORTS ARE DROPPED. The assembler strips each source's top-level imports
    and writes one fixed list in its HEADER. A module the HEADER lacks is
    undefined in `sicopr.py`. Found 2026-10-03: `csv` (read_ParamConfigFile's
    CSV reader, writecsv_transposed) and `io` (read_PR_files) were missing, so
    a .csv config raised NameError in the engine and in no unit test.

  * MODULE NAMES COLLIDE. Private functions are renamed per module, but a
    module-level assignment is not. Two sources that each say
    `_SENTINEL = object()` leave one name bound to the LAST object, while the
    first module's default arguments still hold the first one. Found
    2026-10-03: read_ParamConfigFile's `default_value is _SENTINEL` was never
    true in the engine, so a missing mandatory keyword came back as a bare
    object and failed later with an unrelated TypeError.

What this checks, on `sicopr.py` as assembled:

  1. every name a function reads as a global is bound at module level (an
     import, a def, a class, an assignment, or a `global` assignment inside a
     function) or is a builtin;
  2. no module-level name is bound twice with different source text, or twice
     to a fresh identity object (`object()`). Rebinding the same value with the
     same text (`_EPS = np.finfo(float).eps` in three modules) is harmless and
     allowed.

    python tests/test_assembled_names.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import ast
import builtins
import collections
import io
import os
import sys
import symtable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_check import check, finish  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, 'sicopr.py')

_DUNDERS = {'__file__', '__name__', '__doc__', '__spec__', '__loader__',
            '__package__', '__builtins__', '__path__', '__cached__'}


def _module_bindings(top):
    """Names bound at module level, including by `global x; x = ...` inside a
    function, which binds the module name at run time."""
    bound = set()
    for s in top.get_symbols():
        if s.is_assigned() or s.is_imported() or s.is_namespace():
            bound.add(s.get_name())

    def walk(t):
        for s in t.get_symbols():
            if s.is_declared_global() and s.is_assigned():
                bound.add(s.get_name())
        for c in t.get_children():
            walk(c)
    for c in top.get_children():
        walk(c)
    return bound


def undefined_globals(src, filename='sicopr.py'):
    """{name: [function, ...]} for globals read but never bound."""
    top = symtable.symtable(src, filename, 'exec')
    known = _module_bindings(top) | set(dir(builtins)) | _DUNDERS
    missing = collections.defaultdict(list)

    def walk(t):
        if t.get_type() != 'module':
            for s in t.get_symbols():
                if s.is_global() and s.is_referenced() and s.get_name() not in known:
                    missing[s.get_name()].append(t.get_name())
        for c in t.get_children():
            walk(c)
    walk(top)
    return dict(missing)


def _makes_identity(node):
    """True if the value is a fresh object compared by identity: object()."""
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == 'object' for n in ast.walk(node))


def colliding_bindings(src):
    """{name: [(line, text), ...]} for module-level names bound more than once
    in a way that changes what the name means."""
    tree = ast.parse(src)
    seen = collections.defaultdict(list)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[n.name].append((n.lineno, ast.unparse(n), False))
        elif isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name):
                    seen[tg.id].append((n.lineno, ast.unparse(n.value),
                                        _makes_identity(n.value)))
    bad = {}
    for name, binds in seen.items():
        if len(binds) < 2:
            continue
        if len({b[1] for b in binds}) > 1 or any(b[2] for b in binds):
            bad[name] = [(line, text.splitlines()[0][:70]) for line, text, _ in binds]
    return bad


def main():
    src = io.open(ENGINE, encoding='utf-8').read()

    missing = undefined_globals(src)
    for name, fns in sorted(missing.items()):
        print('  undefined global %r used by %s' % (name, ', '.join(sorted(set(fns)))))
    check('every_global_the_engine_reads_is_bound', not missing,
          '%d undefined: %s' % (len(missing), ', '.join(sorted(missing))))

    clashes = colliding_bindings(src)
    for name, binds in sorted(clashes.items()):
        print('  %r bound %d times:' % (name, len(binds)))
        for line, text in binds:
            print('      sicopr.py:%d  %s' % (line, text))
    check('no_module_name_is_rebound_to_something_else', not clashes,
          '%d names: %s' % (len(clashes), ', '.join(sorted(clashes))))

    # The detectors themselves, on small sources with the two defects in them.
    check('detector_finds_a_dropped_import',
          undefined_globals('def f(p):\n    return csv.reader(p)\n') == {'csv': ['f']})
    check('detector_accepts_an_import',
          undefined_globals('import csv\ndef f(p):\n    return csv.reader(p)\n') == {})
    check('detector_accepts_a_global_assigned_in_a_function',
          undefined_globals('def f():\n    global X\n    X = 1\ndef g():\n    return X\n') == {})
    check('detector_finds_a_sentinel_bound_twice',
          set(colliding_bindings('S = object()\ndef f(d=S):\n    return d is S\nS = object()\n'))
          == {'S'})
    check('detector_allows_the_same_value_twice',
          colliding_bindings('E = 1e-16\nE = 1e-16\n') == {})
    check('detector_finds_a_name_given_two_values',
          set(colliding_bindings('E = 1\nE = 2\n')) == {'E'})
    finish()


if __name__ == '__main__':
    main()
