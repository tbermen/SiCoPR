"""pytest plugin: record, per assertion call site, whether it compared complex
values.

The static complex census in translation_coverage.py sees a complex comparison
only when a complex literal or `.imag` sits inside the assertion itself, so
`assert_allclose(got, _OCT_S21)` against a complex table defined elsewhere is
invisible to it. numpy compares complex operands on the complex difference,
real and imaginary parts together, so an operand's dtype at run time settles
the question the source cannot.

    python -m pytest -p record_comparisons com_functions/fn -q
    (with tools/ on sys.path; RECORD_COMPARISONS=<file.csv> sets the output)

Wraps numpy.testing.assert_allclose / assert_array_equal /
assert_array_almost_equal, numpy.allclose / isclose / array_equal, and
pytest.approx's comparison. Records only call sites inside test files.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import atexit
import csv
import functools
import os
import sys

import numpy as np

_SEEN = {}          # (file, line) -> [calls, complex_calls]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_complex(x):
    try:
        return np.iscomplexobj(np.asarray(x))
    except Exception:                                       # noqa: BLE001
        return isinstance(x, complex)


def _site():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        base = os.path.basename(fn)
        if base.startswith('test_') and fn.endswith('.py'):
            return os.path.relpath(fn, _ROOT).replace('\\', '/'), f.f_lineno
        f = f.f_back
    return None


def _note(a, b):
    s = _site()
    if s is None:
        return
    rec = _SEEN.setdefault(s, [0, 0])
    rec[0] += 1
    if _is_complex(a) or _is_complex(b):
        rec[1] += 1


def _wrap(mod, name):
    orig = getattr(mod, name)

    @functools.wraps(orig)
    def inner(a, b, *args, **kw):
        _note(a, b)
        return orig(a, b, *args, **kw)
    setattr(mod, name, inner)


for _name in ('assert_allclose', 'assert_array_equal',
              'assert_array_almost_equal'):
    _wrap(np.testing, _name)
for _name in ('allclose', 'isclose', 'array_equal'):
    _wrap(np, _name)

try:
    from _pytest import python_api as _pa
    _orig_eq = _pa.ApproxBase.__eq__

    def _approx_eq(self, actual):
        _note(actual, getattr(self, 'expected', None))
        return _orig_eq(self, actual)
    _pa.ApproxBase.__eq__ = _approx_eq
    for _cls in ('ApproxScalar', 'ApproxNumpy', 'ApproxSequenceLike',
                 'ApproxMapping', 'ApproxDecimal'):
        _c = getattr(_pa, _cls, None)
        if _c is not None and '__eq__' in _c.__dict__:
            _o = _c.__dict__['__eq__']

            def _mk(o):
                def _eq(self, actual):
                    _note(actual, getattr(self, 'expected', None))
                    return o(self, actual)
                return _eq
            setattr(_c, '__eq__', _mk(_o))
except Exception:                                           # noqa: BLE001
    pass


@atexit.register
def _dump():
    out = os.environ.get('RECORD_COMPARISONS')
    if not out:
        return
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'line', 'calls', 'complex_calls'])
        for (fn, ln), (n, c) in sorted(_SEEN.items()):
            w.writerow([fn, ln, n, c])
