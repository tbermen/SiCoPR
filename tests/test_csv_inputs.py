"""The engine reads every config format the Reference Code reads.

`read_ParamConfigFile` in the Reference Code takes a workbook, a `.csv`
(`xlsread` on the file, 4p17p0 L10622) or a `.mat` holding the `parameter`
cell array (`load`, L10620). Each is the same grid of keyword and value cells,
and everything after the file reader is shared, so a config must load to the
same `param` and `OP` whichever form it arrives in.

The per-function tests could not see the failures this guards, because they
import `com_functions/fn/*/py_impl.py` directly. These checks go through the
assembled `sicopr`:

  * a `.csv` config raised NameError: `csv` was not imported in the engine
    (finding F01, 2026-10-03);
  * a `.csv` pulse response raised NameError on `io` (F02);
  * writecsv_transposed raised NameError on `csv` (F03);
  * a config missing a mandatory keyword gave an unrelated TypeError instead
    of the missing-keyword error: two modules' `_SENTINEL` collided (F05);
  * the apparent-channel-bandwidth helpers (4p17p0, ACBW = 1) raised NameError
    on `cho_factor`, `cho_solve` and `solve_triangular` (F13).

Every workbook under benchmark/ and examples/ ships with the repository, so
nothing here skips.

    python tests/test_csv_inputs.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import csv
import glob
import io
import math
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from audit_check import check, finish  # noqa: E402

_stdout = sys.stdout
sys.stdout = io.StringIO()          # importing the engine prints its banner
try:
    import sicopr  # noqa: E402
finally:
    sys.stdout = _stdout


def _bootstrap_OP():
    """The OP the driver hands to read_ParamConfigFile, lifted from the driver
    so it cannot drift (as tests/test_config_roundtrip.py does; that file runs
    its checks at import, so it cannot be imported from)."""
    src = io.open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read().splitlines()
    a = next(i for i, l in enumerate(src) if l.strip() == 'OP = SimpleNamespace()')
    b = next(i for i, l in enumerate(src) if 'param, OP = read_ParamConfigFile' in l)
    ns = {'SimpleNamespace': SimpleNamespace}
    exec(chr(10).join(l[4:] for l in src[a:b]), ns)
    return ns['OP']


def _equal(x, y, path, bad):
    """Deep exact equality over namespaces, dicts, lists and arrays."""
    if isinstance(x, SimpleNamespace) or isinstance(y, SimpleNamespace):
        if not (isinstance(x, SimpleNamespace) and isinstance(y, SimpleNamespace)):
            bad.append('%s: %s vs %s' % (path, type(x).__name__, type(y).__name__))
        else:
            _diff(vars(x), vars(y), path, bad)
        return
    if isinstance(x, dict) and isinstance(y, dict):
        _diff(x, y, path, bad)
        return
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            bad.append('%s: length %d vs %d' % (path, len(x), len(y)))
            return
        for i, (u, v) in enumerate(zip(x, y)):
            _equal(u, v, '%s[%d]' % (path, i), bad)
        return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            bad.append('%s: shape %s vs %s' % (path, x.shape, y.shape))
        elif x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
            if not np.array_equal(x, y, equal_nan=True):
                bad.append('%s: values differ' % path)
        elif not np.array_equal(x, y):
            bad.append('%s: values differ' % path)
        return
    if isinstance(x, float) and isinstance(y, float):
        if not (x == y or (math.isnan(x) and math.isnan(y))):
            bad.append('%s: %r vs %r' % (path, x, y))
        return
    if type(x) is not type(y) and not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        bad.append('%s: type %s vs %s' % (path, type(x).__name__, type(y).__name__))
        return
    if x is not y and x != y:
        bad.append('%s: %r vs %r' % (path, x, y))


def _diff(a, b, label, bad):
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append('%s.%s present in only one' % (label, k))
            continue
        try:
            _equal(a[k], b[k], '%s.%s' % (label, k), bad)
        except Exception as e:                       # noqa: BLE001
            bad.append('%s.%s comparison failed: %s' % (label, k, e))
    return bad


def _workbooks():
    out = []
    for d in ('benchmark', 'examples'):
        out += glob.glob(os.path.join(_ROOT, d, '**', '*.xlsx'), recursive=True)
    return sorted(p for p in out if not os.path.basename(p).startswith('~$'))


def _load(path):
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        param, OP = sicopr.read_ParamConfigFile(path, _bootstrap_OP())
    finally:
        sys.stdout = old
    return vars(param), vars(OP)


def _grid(path):
    """COM_Settings as xlsread's raw output holds it, from the converter the
    Octave route uses, so the .csv and the .mat carry the same grid."""
    from xlsx_to_com_mat import read_grid, trim
    return trim(read_grid(path))


def _csv_text(cell):
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return ''
    if isinstance(cell, bool):
        return '1' if cell else '0'
    if isinstance(cell, float):
        return repr(cell)
    return str(cell)


def _write_csv(grid, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for row in grid:
            w.writerow([_csv_text(c) for c in row])


def _fields_equal(a, b, label):
    pa, oa = a
    pb, ob = b
    return _diff(pa, pb, label + '.param', []) + _diff(oa, ob, label + '.OP', [])


def main():
    books = _workbooks()
    print('%d workbooks under benchmark/ and examples/' % len(books))
    check('there_are_workbooks_to_compare', len(books) >= 9, '%d found' % len(books))
    tmp = tempfile.mkdtemp(prefix='sicopr_csv_')
    try:
        n_csv = 0
        for i, src in enumerate(books):
            name = os.path.basename(src)
            # A load can write a .mat beside the file it reads; work on copies.
            xlsx = os.path.join(tmp, '%02d_%s' % (i, name))
            shutil.copyfile(src, xlsx)
            want = _load(xlsx)
            grid = _grid(xlsx)

            path = os.path.join(tmp, '%02d.csv' % i)
            _write_csv(grid, path)
            try:
                bad = _fields_equal(want, _load(path), name)
            except Exception as e:                      # noqa: BLE001
                bad = ['%s: %s' % (type(e).__name__, e)]
            n_csv += not bad
            check('csv_loads_like_the_workbook: %s' % name, not bad, '; '.join(bad[:3]))

        print('compared %d workbooks: %d .csv loaded identically' % (len(books), n_csv))

        # F05: a missing mandatory keyword is the missing-keyword error.
        grid = _grid(os.path.join(tmp, '00_' + os.path.basename(books[0])))
        path = os.path.join(tmp, 'no_M.csv')
        _write_csv([r for r in grid if not (r and isinstance(r[0], str)
                                            and r[0].strip().lower() == 'm')], path)
        try:
            _load(path)
            err = None
        except Exception as e:                          # noqa: BLE001
            err = e
        check('a_missing_mandatory_keyword_names_it',
              isinstance(err, KeyError) and 'Mandatory parameter "M"' in str(err),
              repr(err))

        # F02: a .csv pulse response, read as the per-function code reads it.
        from com_functions.fn.read_PR_files.py_impl import read_PR_files as ref_read
        t = np.arange(40) * 1e-11
        v = np.exp(-((t - 1.5e-10) / 5e-11) ** 2)
        path = os.path.join(tmp, 'pr.csv')
        with open(path, 'w') as f:
            for a, b in zip(t, v):
                f.write('%r,%r\n' % (float(a), float(b)))

        def _ch():
            return [SimpleNamespace(ext='.csv', filename=path)]
        param = SimpleNamespace(samples_per_ui=8)
        try:
            got, _ = sicopr.read_PR_files(param, SimpleNamespace(), _ch())
            exp, _ = ref_read(param, SimpleNamespace(), _ch())
            ok = (np.array_equal(got[0].uneq_pulse_response, exp[0].uneq_pulse_response)
                  and np.array_equal(got[0].uneq_imp_response, exp[0].uneq_imp_response))
            why = ''
        except Exception as e:                          # noqa: BLE001
            ok, why = False, repr(e)
        check('the_engine_reads_a_csv_pulse_response', ok, why)

        # F03: writecsv_transposed.
        from com_functions.fn.writecsv_transposed.py_impl import writecsv_transposed as ref_w
        ns = SimpleNamespace(COM=3.25, name='case', taps=np.array([0.1, -0.2]))
        a, b = os.path.join(tmp, 'w1.csv'), os.path.join(tmp, 'w2.csv')
        try:
            sicopr.writecsv_transposed(ns, a)
            ref_w(ns, b)
            ok = open(a).read() == open(b).read()
            why = ''
        except Exception as e:                          # noqa: BLE001
            ok, why = False, repr(e)
        check('the_engine_writes_a_transposed_csv', ok, why)

        # F13: the ACBW helpers, on the input of get_CICP_fit_sweep's own test.
        from com_functions.fn.get_CICP_fit_sweep.py_impl import get_CICP_fit_sweep as ref_s
        f = np.linspace(0.5, 100, 400)
        y = -40 + 8 * np.log10(f) + 0.002 * f ** 1.8
        kw = dict(f_fit_min_GHz=10, f_upper_min_GHz=20, f_upper_max_GHz=60,
                  f_test_min_GHz=20, f_test_max_GHz=60, step_GHz=10)
        try:
            got = sicopr.get_CICP_fit_sweep(y, f, SimpleNamespace(DISPLAY_WINDOW=0), **kw)
            exp = ref_s(y, f, SimpleNamespace(DISPLAY_WINDOW=0), **kw)
            ok = all(np.array_equal(np.asarray(g), np.asarray(e)) for g, e in zip(got, exp))
            why = ''
        except Exception as e:                          # noqa: BLE001
            ok, why = False, repr(e)
        check('the_engine_runs_the_acbw_fit', ok, why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    finish()


if __name__ == '__main__':
    main()
