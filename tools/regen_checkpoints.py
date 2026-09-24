"""regen_checkpoints.py -- the Octave checkpoint goldens, one directory per case.

    python tools/regen_checkpoints.py --cases CASES.json [--cases MORE.json]
                                      [--version 4p15p0|4p16p0] [--out DIR]
                                      [--jobs 3] [--only ID,ID]

Runs the reference, through its Octave-capable release file, with
COM_CHECKPOINT_DIR set, so octave/patches/com_checkpoint.m saves every struct
the pipeline holds at ten stage boundaries of main. What it writes:

    <out>/<version>/<case id>/NN_<stage>_pc<k>[_n].mat   the checkpoints
    <out>/<version>/<case id>/results.mat                main's return value
    <out>/<version>/manifest.json                        what produced them

A case is {"id", "config", "thru", "fext": [...], "next": [...]}, the shape
tools/octave_compare.py already reads. The configuration is converted to the
.mat parameter array with RESULT_DIR, SAVE_FIGURES=0, DISPLAY_WINDOW=0 and
CSV_REPORT=1 changed and nothing else, exactly as octave_compare.py does.

<out> defaults to $COM_OCTAVE_CHECKPOINTS. The goldens are derived from the
channels, so they carry the channels' redistribution terms: for the IEEE
public-area set that means they stay out of every repository, which is why the
output directory is an argument rather than a path inside this one.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octave_compare import OCTAVE_DIR, convert, find_octave   # noqa: E402

VERSIONS = ('4p15p0', '4p16p0')


def _fwd(p):
    return os.path.abspath(p).replace('\\', '/')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def compat_file(version):
    return os.path.join(OCTAVE_DIR, 'com_ieee8023_%s_octave_compat.m' % version)


def source_sha(version):
    """The sha256 of the MATLAB release the compat file was generated from."""
    with open(compat_file(version), encoding='latin-1') as fh:
        head = fh.read(4000)
    m = re.search(r'sha256 ([0-9a-f]{64})', head)
    return m.group(1) if m else ''


def octave_version(octave):
    p = subprocess.run([octave, '--version'], capture_output=True, text=True,
                       errors='replace')
    return (p.stdout.strip().split('\n') or [''])[0]


def run_case(case, version, out_root, octave, timeout):
    wd = os.path.join(out_root, case['id'])
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    scratch = os.path.join(wd, '_run')
    os.makedirs(os.path.join(scratch, 'octave_results'))
    mat = os.path.join(scratch, 'config.mat')
    convert(case['config'], mat,
            [('RESULT_DIR', _fwd(os.path.join(scratch, 'octave_results')) + '/'),
             ('SAVE_FIGURES', '0'), ('DISPLAY_WINDOW', '0'), ('CSV_REPORT', '1')])
    files = [case['thru']] + list(case.get('fext', [])) + list(case.get('next', []))
    args = ["'%s'" % _fwd(mat), str(len(case.get('fext', []))),
            str(len(case.get('next', [])))] + ["'%s'" % _fwd(f) for f in files]
    results = os.path.join(wd, 'results.mat')
    ev = ("more off; addpath('%s'); r = com_ieee8023_%s_octave_compat(%s); "
          "save('-v7','%s','r'); exit(0);"
          % (_fwd(OCTAVE_DIR), version, ', '.join(args), _fwd(results)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               COM_CHECKPOINT_DIR=os.path.abspath(wd))
    t0 = time.time()
    p = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       capture_output=True, text=True, cwd=scratch, env=env,
                       timeout=timeout, errors='replace')
    wall = time.time() - t0
    with open(os.path.join(scratch, 'octave.log'), 'w', encoding='utf-8',
              errors='replace') as fh:
        fh.write('=== eval ===\n%s\n=== exit %s, wall %.1fs ===\n%s\n%s\n'
                 % (ev, p.returncode, wall, p.stdout, p.stderr))
    stages = sorted(f for f in os.listdir(wd)
                    if f.endswith('.mat') and f != 'results.mat')
    failed_saves = re.findall(r'com_checkpoint: .*', p.stderr)
    ok = p.returncode == 0 and os.path.isfile(results) and not failed_saves
    return {'id': case['id'], 'config': case['config'],
            'config_sha256': _sha256(case['config']),
            'thru': case['thru'], 'fext': list(case.get('fext', [])),
            'next': list(case.get('next', [])), 'stages': stages,
            'ok': ok, 'exit': p.returncode, 'wall_s': round(wall, 1),
            'failed_saves': failed_saves}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cases', action='append', required=True,
                    help='a JSON list of cases; may be given more than once')
    ap.add_argument('--version', choices=VERSIONS, default='4p15p0',
                    help='the release to run (default 4p15p0, the build the '
                         'MATLAB reference results were produced with)')
    ap.add_argument('--out', default=os.environ.get('COM_OCTAVE_CHECKPOINTS'),
                    help='output root (default $COM_OCTAVE_CHECKPOINTS)')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--only', help='comma-separated case ids')
    ap.add_argument('--timeout', type=int, default=4 * 3600)
    a = ap.parse_args(argv)
    if not a.out:
        ap.error('no --out and COM_OCTAVE_CHECKPOINTS is not set')
    octave = find_octave()
    if not octave:
        ap.error('octave-cli not found')

    cases = []
    for path in a.cases:
        with open(path, encoding='utf-8') as fh:
            cases += json.load(fh)
    if a.only:
        want = set(a.only.split(','))
        cases = [c for c in cases if c['id'] in want]
    out_root = os.path.join(a.out, a.version)
    os.makedirs(out_root, exist_ok=True)
    man_path = os.path.join(out_root, 'manifest.json')
    manifest = {'cases': {}}
    if os.path.isfile(man_path):
        with open(man_path, encoding='utf-8') as fh:
            manifest = json.load(fh)
    manifest.update({
        'written': datetime.datetime.now().isoformat(timespec='seconds'),
        'version': a.version,
        'octave': octave_version(octave),
        'compat_file': os.path.basename(compat_file(a.version)),
        'compat_sha256': _sha256(compat_file(a.version)),
        'release_sha256': source_sha(a.version),
        'accel': os.environ.get('COM_OCTAVE_ACCEL', '') != '0',
    })

    # the with-crosstalk cases sweep the Tx FFE and run far longer; start them
    # first so they overlap the short ones instead of trailing alone
    cases.sort(key=lambda c: -(len(c.get('fext', [])) + len(c.get('next', []))))
    print('%d case(s), %s, %d at a time -> %s' % (len(cases), a.version,
                                                   a.jobs, out_root))
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_case, c, a.version, out_root, octave, a.timeout): c
                for c in cases}
        for i, f in enumerate(cf.as_completed(futs), 1):
            c = futs[f]
            try:
                r = f.result()
            except Exception as e:                              # noqa: BLE001
                r = {'id': c['id'], 'ok': False, 'error': str(e)[:300]}
            manifest['cases'][r['id']] = r
            with open(man_path, 'w', encoding='utf-8') as fh:
                json.dump(manifest, fh, indent=1)
            print('[%d/%d] %-18s %s  %d stage file(s)  %.1f min   (elapsed %.2f h)'
                  % (i, len(cases), r['id'], 'ok    ' if r.get('ok') else 'FAILED',
                     len(r.get('stages', [])), r.get('wall_s', 0) / 60,
                     (time.time() - t0) / 3600), flush=True)
    bad = [k for k, v in manifest['cases'].items() if not v.get('ok')]
    print('done: %d ok, %d failed%s' % (len(manifest['cases']) - len(bad), len(bad),
                                        (': ' + ', '.join(bad)) if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
