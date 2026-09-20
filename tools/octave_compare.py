"""octave_compare.py -- the same COM case through GNU Octave and SiCoPR, compared.

    python tools/octave_compare.py CONFIG.xlsx THRU.s4p [--fext F.s4p ...] [--next N.s4p ...]
                                   [--version 4p15p0|4p16p0] [--out DIR]
    python tools/octave_compare.py --cases cases.json [--version V] [--out DIR] [--jobs N]

Two implementations of the same reference are worth more than one when they
can be run side by side on demand. The Octave side is the release file made
Octave-capable by octave/make_octave_compat.py; the Python side is `python -m
sicopr`, emulating the same release. Both live in this repository, so anyone
with a workbook and a channel can reproduce a row of the comparison without a
MATLAB licence.

What is compared: the scalars COM reports and both engines write (COM_dB, FOM,
the sampling phase, ERL, VEC, VEO, ICN, the DER terms, the CTLE gain) and the
equalizer vectors (Tx FFE, DFE taps). Deltas are Octave minus SiCoPR.

Batch mode reads a JSON list of cases

    [{"id": "woXtalk_T1_R03", "config": "...xlsx", "thru": "...s4p",
      "fext": ["...s4p"], "next": []}, ...]

runs them `--jobs` at a time and writes `<out>/compare.csv` plus a summary.
Octave is loop-bound and light on memory, so several fit on one machine; each
gets a single BLAS thread so N workers do not each spin up 16.

Every run leaves its evidence under `<out>/<id>/`: the converted config, the
Octave log and result, and the SiCoPR results tree. Nothing is cleaned up.

The Octave side runs about 1.6x faster if octave/com_octave_accel.oct has been
built (python octave/accel/build_accel.py); it is picked up automatically, needs
no flag here, and returns the same bits. COM_OCTAVE_ACCEL=0 runs without it.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import concurrent.futures
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from xlsx_to_com_mat import convert, find_octave  # noqa: E402

OCTAVE_DIR = os.path.join(_ROOT, 'octave')
VERSIONS = ('4p15p0', '4p16p0')

SCALARS = ['COM_dB', 'FOM', 'itick', 'ERL', 'VEC_dB', 'VEO_mV', 'ICN_mV',
           'DER_thresh', 'DER_MLSE', 'DER_DFE', 'CTLE_DC_gain_dB',
           'channel_operating_margin_dB']
VECTORS = ['TXLE_taps', 'DFE_taps']
EXACT = 1e-6         # the three-way study's EXACT class on COM_dB


def _fwd(p):
    return os.path.abspath(p).replace('\\', '/')


# ------------------------------------------------------------------ octave

def run_octave(version, config, thru, fext, nxt, workdir, octave=None, timeout=7200):
    """One case under Octave. -> dict of scalars/vectors plus 'wall_s', 'log'."""
    octave = octave or find_octave()
    if not octave:
        raise RuntimeError('octave-cli not found; give --octave')
    if version not in VERSIONS:
        raise ValueError('version must be one of %s' % (VERSIONS,))
    os.makedirs(workdir, exist_ok=True)
    result_dir = os.path.join(workdir, 'octave_results') + os.sep
    os.makedirs(result_dir, exist_ok=True)
    mat = os.path.join(workdir, 'config.mat')
    # Headless, and the report goes where we can find it. Nothing else in the
    # workbook is touched; in particular the search settings stay as written.
    convert(config, mat, [('RESULT_DIR', _fwd(result_dir) + '/'),
                          ('SAVE_FIGURES', '0'), ('DISPLAY_WINDOW', '0'),
                          ('CSV_REPORT', '1')])
    out_mat = os.path.join(workdir, 'octave_result.mat')
    if os.path.exists(out_mat):
        os.remove(out_mat)
    files = [thru] + list(fext) + list(nxt)
    args = ["'%s'" % _fwd(mat), str(len(fext)), str(len(nxt))] + ["'%s'" % _fwd(f) for f in files]
    entry = 'com_ieee8023_%s_octave_compat' % version
    ev = ("more off; addpath('%s'); r = %s(%s); save('-v7','%s','r'); exit(0);"
          % (_fwd(OCTAVE_DIR), entry, ', '.join(args), _fwd(out_mat)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    t0 = time.time()
    p = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       capture_output=True, text=True, cwd=workdir, env=env,
                       timeout=timeout, errors='replace')
    wall = time.time() - t0
    log = os.path.join(workdir, 'octave.log')
    with open(log, 'w', encoding='utf-8', errors='replace') as fh:
        fh.write('=== eval ===\n%s\n=== exit %s, wall %.1fs ===\n=== stdout ===\n%s\n'
                 '=== stderr ===\n%s\n' % (ev, p.returncode, wall, p.stdout, p.stderr))
    if p.returncode != 0 or not os.path.isfile(out_mat):
        err = re.search(r'^\s*error:.*$', p.stdout + '\n' + p.stderr, re.M)
        raise RuntimeError('octave failed (exit %s): %s; see %s'
                           % (p.returncode, err.group(0).strip() if err else 'no error line', log))
    import scipy.io
    r = scipy.io.loadmat(out_mat, squeeze_me=True, struct_as_record=False)['r']
    out = {'wall_s': wall, 'log': log}
    for k in SCALARS:
        v = getattr(r, k, None)
        if v is not None and np.asarray(v).size == 1:
            out[k] = float(np.asarray(v).ravel()[0])
    for k in VECTORS:
        v = getattr(r, k, None)
        if v is not None:
            out[k] = [float(x) for x in np.asarray(v, dtype=float).ravel()]
    return out


# ------------------------------------------------------------------ sicopr

def _vector(text):
    """'[0 0 1 0]' or '0.1' -> list of floats; None when it is not numeric."""
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(text))
    return [float(x) for x in nums] if nums else None


def run_sicopr(version, config, thru, fext, nxt, workdir, timeout=7200):
    """One case through `python -m sicopr`. -> dict like run_octave's."""
    os.makedirs(workdir, exist_ok=True)
    cmd = [sys.executable, '-m', 'sicopr', os.path.abspath(config), os.path.abspath(thru)]
    if fext:
        cmd += ['--fext'] + [os.path.abspath(f) for f in fext]
    if nxt:
        cmd += ['--next'] + [os.path.abspath(f) for f in nxt]
    cmd += ['--matlab-version', version]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir,
                       timeout=timeout, errors='replace')
    wall = time.time() - t0
    log = os.path.join(workdir, 'sicopr.log')
    with open(log, 'w', encoding='utf-8', errors='replace') as fh:
        fh.write('=== cmd ===\n%s\n=== exit %s, wall %.1fs ===\n%s\n%s\n'
                 % (' '.join(cmd), p.returncode, wall, p.stdout, p.stderr))
    hits = sorted(glob.glob(os.path.join(workdir, '**', 'results.csv'), recursive=True),
                  key=os.path.getmtime)
    if p.returncode != 0 or not hits:
        raise RuntimeError('sicopr failed (exit %s); see %s' % (p.returncode, log))
    with open(hits[-1], newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.reader(fh))
    row = dict(zip(rows[0], rows[1]))
    out = {'wall_s': wall, 'log': log, 'results_csv': hits[-1]}
    for k in SCALARS:
        if k in row and str(row[k]).strip() != '':
            try:
                out[k] = float(row[k])
            except ValueError:
                pass
    for k in VECTORS:
        if k in row:
            v = _vector(row[k])
            if v is not None:
                out[k] = v
    return out


# ----------------------------------------------------------------- compare

def compare(oct_r, py_r):
    """Per-key deltas, Octave minus SiCoPR. -> list of (key, oct, py, delta)."""
    rows = []
    for k in SCALARS:
        a, b = oct_r.get(k), py_r.get(k)
        if a is None or b is None:
            continue
        rows.append((k, a, b, a - b))
    for k in VECTORS:
        a, b = oct_r.get(k), py_r.get(k)
        if a is None or b is None:
            continue
        # trailing zeros are not a disagreement about the equalizer
        while len(a) > len(b) and a[-1] == 0:
            a = a[:-1]
        while len(b) > len(a) and b[-1] == 0:
            b = b[:-1]
        d = max((abs(x - y) for x, y in zip(a, b)), default=0.0) if len(a) == len(b) else float('nan')
        rows.append((k, a, b, d))
    return rows


def run_case(case, version, out_dir, octave=None):
    """Both engines on one case. -> flat dict for a CSV row."""
    cid = case.get('id') or os.path.splitext(os.path.basename(case['thru']))[0]
    work = os.path.join(out_dir, cid)
    row = {'id': cid, 'version': version, 'config': os.path.basename(case['config']),
           'thru': os.path.basename(case['thru']),
           'n_fext': len(case.get('fext') or []), 'n_next': len(case.get('next') or [])}
    try:
        oct_r = run_octave(version, case['config'], case['thru'], case.get('fext') or [],
                           case.get('next') or [], os.path.join(work, 'octave'), octave)
        row['octave_wall_s'] = round(oct_r['wall_s'], 1)
    except Exception as e:                                       # noqa: BLE001
        row['octave_error'] = str(e)[:300]
        oct_r = {}
    try:
        py_r = run_sicopr(version, case['config'], case['thru'], case.get('fext') or [],
                          case.get('next') or [], os.path.join(work, 'sicopr'))
        row['sicopr_wall_s'] = round(py_r['wall_s'], 1)
    except Exception as e:                                       # noqa: BLE001
        row['sicopr_error'] = str(e)[:300]
        py_r = {}
    for k, a, b, d in compare(oct_r, py_r):
        row[k + '_oct'] = a if not isinstance(a, list) else ' '.join('%.10g' % x for x in a)
        row[k + '_py'] = b if not isinstance(b, list) else ' '.join('%.10g' % x for x in b)
        row['d_' + k] = d
    return row


def print_table(row):
    print('%-30s %22s %22s %14s' % ('field', 'Octave', 'SiCoPR', 'Octave - SiCoPR'))
    for k in SCALARS + VECTORS:
        if k + '_oct' not in row:
            continue
        a, b, d = row[k + '_oct'], row[k + '_py'], row['d_' + k]
        fa = a if isinstance(a, str) else '%.15g' % a
        fb = b if isinstance(b, str) else '%.15g' % b
        print('%-30s %22s %22s %14.3e' % (k, fa[:22], fb[:22], d))
    print('\nOctave %.1f s, SiCoPR %.1f s (ratio %.2fx)'
          % (row.get('octave_wall_s', float('nan')), row.get('sicopr_wall_s', float('nan')),
             row.get('octave_wall_s', float('nan')) / max(row.get('sicopr_wall_s', 1e-9), 1e-9)))
    for k in ('octave_error', 'sicopr_error'):
        if k in row:
            print('%s: %s' % (k, row[k]))


def summarise(rows):
    d = [abs(r['d_COM_dB']) for r in rows if 'd_COM_dB' in r and np.isfinite(r['d_COM_dB'])]
    n_err = sum(1 for r in rows if 'octave_error' in r or 'sicopr_error' in r)
    s = {'cases': len(rows), 'compared': len(d), 'errors': n_err,
         'COM_exact_1e-6': sum(1 for x in d if x <= EXACT),
         'max_abs_dCOM_dB': max(d) if d else None,
         'median_abs_dCOM_dB': float(np.median(d)) if d else None,
         'itick_match': sum(1 for r in rows if r.get('d_itick') == 0),
         'octave_total_s': sum(r.get('octave_wall_s', 0) for r in rows),
         'sicopr_total_s': sum(r.get('sicopr_wall_s', 0) for r in rows)}
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('config', nargs='?', help='COM configuration workbook (.xlsx)')
    ap.add_argument('thru', nargs='?', help='THRU (victim) .s4p')
    ap.add_argument('--fext', nargs='*', default=[])
    ap.add_argument('--next', nargs='*', default=[], dest='nxt')
    ap.add_argument('--cases', help='JSON list of cases for batch mode')
    ap.add_argument('--version', default='4p15p0', choices=VERSIONS)
    ap.add_argument('--out', default='octave_compare_out')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--octave', help='path to octave-cli (default: PATH, then the stock Windows install)')
    a = ap.parse_args(argv)

    if a.cases:
        cases = json.load(open(a.cases, encoding='utf-8'))
    elif a.config and a.thru:
        cases = [{'id': os.path.splitext(os.path.basename(a.thru))[0],
                  'config': a.config, 'thru': a.thru, 'fext': a.fext, 'next': a.nxt}]
    else:
        ap.error('give CONFIG and THRU, or --cases')
    os.makedirs(a.out, exist_ok=True)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(run_case, c, a.version, a.out, a.octave): c for c in cases}
        for f in concurrent.futures.as_completed(futs):
            row = f.result()
            rows.append(row)
            d = row.get('d_COM_dB')
            print('%-24s dCOM %s  octave %ss  sicopr %ss%s'
                  % (row['id'], ('%.3e' % d) if d is not None else 'n/a',
                     row.get('octave_wall_s', '?'), row.get('sicopr_wall_s', '?'),
                     ('  ERROR ' + (row.get('octave_error') or row.get('sicopr_error', '')))
                     if 'octave_error' in row or 'sicopr_error' in row else ''), flush=True)
    rows.sort(key=lambda r: r['id'])

    if len(rows) == 1 and not a.cases:
        print()
        print_table(rows[0])
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out_csv = os.path.join(a.out, 'compare.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    s = summarise(rows)
    with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(s, fh, indent=1)
    print('\n%d case(s), %d compared, %d error(s); COM within %g dB on %d; max |dCOM| %s dB; '
          'Octave %.0f s vs SiCoPR %.0f s\n-> %s'
          % (s['cases'], s['compared'], s['errors'], EXACT, s['COM_exact_1e-6'],
             ('%.3e' % s['max_abs_dCOM_dB']) if s['max_abs_dCOM_dB'] is not None else 'n/a',
             s['octave_total_s'], s['sicopr_total_s'], out_csv))
    return 0 if s['errors'] == 0 and s['compared'] == s['COM_exact_1e-6'] else 1


if __name__ == '__main__':
    sys.exit(main())
