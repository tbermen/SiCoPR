"""Is a candidate engine equivalent to the baseline? The owner's rule, 2026-09-24.

A speed-up is kept only if, on every checkpoint case run for both engines:

  STRICT     every reported output (output_args) that is not a statistic of
             the noise distribution, and every fom_result field outside its
             noise PSDs -- COM, VEC, VEO, FOM, the sampling point, every EQ
             setting and tap, ERL, the channel figures -- is BIT-IDENTICAL;
  DER        the detector error ratios (DER_DFE, DER_MLSE, DER_thresh /
             threshold_DER), read off the noise-CDF tail, move by at most
             REL_TOL relative -- unless the candidate's checkpoint table marks
             the row EDGE, with its proof, in which case a bin flip is allowed;
  NOISE      every other saved field -- the noise PDFs and CDFs and the
             statistics taken from them (sgm_*, sigma_*, *noise*), PSD_results,
             the COM_SNR_Struct internals -- moves by at most REL_TOL relative,
             element by element, with its zero pattern unchanged. Element by
             element is the point: it is the far tail that an FFT convolution
             destroyed while every peak-relative comparison passed;
  OCTAVE     agreement with COM Octave is no worse on any field: no status
             becomes FAIL, and no field's worst relative error rises by more
             than OCTAVE_SLACK (a couple of ulp).

Inputs are two report directories written by tests/test_octave_checkpoints.py
(COM_CHECKPOINT_REPORT), one per engine; each holds <case>/checkpoints/*.pkl
and <case>/fields.csv.

    python tools/equivalence_check.py BASE_REPORT CAND_REPORT [case ...]

Exit status 0 when every case is equivalent.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import csv
import importlib.util
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL_TOL = 1e-12
OCTAVE_SLACK = 4e-15
DER = re.compile(r'(^|\.)(DER_DFE|DER_MLSE|DER_thresh|threshold_DER)$')
# output_args fields that are statistics of the noise distribution
NOISE_STAT = re.compile(r'(^|\.)(sgm_|sigma_)|noise|pdf|cdf', re.IGNORECASE)


def _ck():
    spec = importlib.util.spec_from_file_location(
        '_ckpt', os.path.join(ROOT, 'tests', 'test_octave_checkpoints.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CK = _ck()


def klass(path):
    """path like 'output_args.COM_dB' or 'fom_result.PSD_results.S_xn'."""
    canon = CK._canon(path)
    if canon in CK.ENVIRONMENT or path.startswith('OP.export_'):
        return 'ENV'
    if DER.search(canon):
        return 'DER'
    if canon.startswith('output_args.'):
        return 'NOISE' if NOISE_STAT.search(canon[len('output_args.'):]) else 'STRICT'
    if canon.startswith('fom_result.') and not canon.startswith('fom_result.PSD_results'):
        return 'STRICT'
    return 'NOISE'


def rel(a, b):
    """max elementwise relative distance from the baseline; inf when the
    NaN/Inf or zero pattern differs."""
    a = np.asarray(a, dtype=complex).ravel()
    b = np.asarray(b, dtype=complex).ravel()
    worst = 0.0
    for x, y in ((a.real, b.real), (a.imag, b.imag)):
        f = np.isfinite(x) & np.isfinite(y)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return np.inf
        if not np.array_equal(x[~f & ~np.isnan(x)], y[~f & ~np.isnan(y)]):
            return np.inf
        if not np.array_equal(x[f] == 0, y[f] == 0):
            return np.inf                         # the zero pattern moved
        d = np.abs(x[f] - y[f])
        nz = x[f] != 0
        if d.size and d.max() > 0:
            worst = max(worst, float(np.max(d[nz] / np.abs(x[f][nz]))))
    return worst


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            p = (path + '.' + k) if path else k
            if k not in a or k not in b:
                out.append((p, 'present in one engine only'))
            else:
                walk(a[k], b[k], p, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, 'length %d vs %d' % (len(a), len(b))))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, '%s[%d]' % (path, i), out)
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape:
            out.append((path, 'shape %s vs %s' % (x.shape, y.shape)))
        elif x.dtype.kind in 'biufc' and y.dtype.kind in 'biufc':
            if not np.array_equal(x, y, equal_nan=True):
                out.append((path, rel(x, y)))
        elif not np.array_equal(x, y):
            out.append((path, 'values differ'))
        return
    if a is b or a == b or (isinstance(a, float) and a != a and b != b):
        return
    if isinstance(a, (int, float, complex)) and isinstance(b, (int, float, complex)):
        out.append((path, rel(a, b)))
    else:
        out.append((path, 'values differ'))


def fields(report, cid):
    p = os.path.join(report, cid, 'fields.csv')
    if not os.path.isfile(p):
        return {}
    with open(p, encoding='utf-8') as fh:
        return {(r['checkpoint'], r['field']): r for r in csv.DictReader(fh)}


def check_case(base, cand, cid):
    bad, moved = [], []
    bck, cck = (os.path.join(r, cid, 'checkpoints') for r in (base, cand))
    for f in sorted(os.listdir(bck)):
        if not f.endswith('.pkl'):
            continue
        cf = os.path.join(cck, f)
        if not os.path.isfile(cf):
            bad.append((f[:-4], 'checkpoint missing in candidate'))
            continue
        a, _ = CK._load_python(os.path.join(bck, f))
        b, _ = CK._load_python(cf)
        diffs = []
        walk(a, b, '', diffs)
        for path, what in diffs:
            k = klass(path)
            where = '%s:%s' % (f[:-4], path)
            if k == 'ENV':
                continue
            if isinstance(what, str) or k == 'STRICT':
                bad.append((where, '%s field moved (%s)' % (k, what)))
            elif what > REL_TOL:
                if k == 'DER':
                    moved.append((where, what, 'DER beyond %g' % REL_TOL))
                else:
                    bad.append((where, 'moved %.3g relative (limit %g)' % (what, REL_TOL)))
            else:
                moved.append((where, what, k))
    # DER beyond REL_TOL is allowed only where the candidate's table says EDGE
    cf = fields(cand, cid)
    for where, what, k in list(moved):
        if k.startswith('DER beyond'):
            st, path = where.split(':', 1)
            row = cf.get((st, path))
            if not (row and row['status'] == 'EDGE'):
                bad.append((where, 'DER moved %.3g relative and is not an EDGE row' % what))
    # agreement with COM Octave no worse
    bf = fields(base, cid)
    for key, rb in bf.items():
        rc = cf.get(key)
        if rc is None:
            bad.append(('%s:%s' % key, 'row missing in candidate table'))
            continue
        if rc['status'] == 'FAIL' and rb['status'] != 'FAIL':
            bad.append(('%s:%s' % key, 'now FAILS against Octave'))
        try:
            xb, xc = float(rb['max_rel'] or 0), float(rc['max_rel'] or 0)
        except ValueError:
            continue
        if rc['status'] != 'EDGE' and xc > xb + OCTAVE_SLACK:
            bad.append(('%s:%s' % key, 'Octave agreement worse: %.3g -> %.3g' % (xb, xc)))
    return bad, moved


def main(argv):
    base, cand = argv[0], argv[1]
    for d in (base, cand):
        if not os.path.isdir(d):
            print('%s is not a report directory -- nothing was compared, so '
                  'nothing is shown equivalent' % d)
            return 2
    cases = argv[2:] or sorted(set(os.listdir(base)) & set(os.listdir(cand)))
    cases = [c for c in cases if os.path.isdir(os.path.join(base, c, 'checkpoints'))]
    if not cases:
        print('no case has checkpoints in both %s and %s -- nothing was compared, '
              'so nothing is shown equivalent' % (base, cand))
        return 2
    ok_all = True
    for cid in cases:
        bad, moved = check_case(base, cand, cid)
        worst = max((m[1] for m in moved), default=0)
        print('%-18s %s   fields moved within the rule: %3d, worst %.3g relative'
              % (cid, 'EQUIVALENT' if not bad else 'NOT EQUIVALENT', len(moved), worst))
        for where, why in bad[:8]:
            print('     %-70s %s' % (where[:70], why))
        ok_all &= not bad
    print('\n%d case(s): %s' % (len(cases), 'ALL EQUIVALENT' if ok_all else 'NOT EQUIVALENT'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
