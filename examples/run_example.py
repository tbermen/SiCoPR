"""run_example.py -- run the worked example and check it against the values it ships with.

    python examples/run_example.py --channels DIR                  # both engines, both conditions
    python examples/run_example.py --channels DIR --engine sicopr
    python examples/run_example.py --channels DIR --condition no_crosstalk

DIR is wherever you unpacked the channel files named in the example's CHANNEL.md.
They are an IEEE 802.3 contribution, public but not redistributed here, so this
script starts by checking that the files you fetched are the ones the expected
values were produced from.

What it then does, per engine and condition: runs the case, reads the COM report
it writes, and compares every scalar the example pins against the value shipped
beside the configuration. Anything that differs by more than a whisker is
printed. Nothing is written into the repository; results go to --out (default: a
temporary directory).

The Octave side runs only if octave-cli is on PATH (or --octave is given). It
needs the configuration as a .mat, which this script produces with
tools/xlsx_to_com_mat.py, the same way a hand run would.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import csv
import glob
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

EXAMPLE = 'akinwale_CR_22dB_VendorX'
# Scalars the example pins, and how close a run has to land. COM and FOM are in
# dB; the tolerance is far tighter than anything an engineer would act on, and
# loose enough not to trip on a different BLAS.
TOL = {'COM_dB': 1e-9, 'FOM': 1e-9, 'itick': 0, 'ERL': 1e-9, 'VEC_dB': 1e-9,
       'VEO_mV': 1e-6, 'ICN_mV': 1e-9}


def die(msg):
    sys.exit('run_example: %s' % msg)


def sha256(p):
    h = hashlib.sha256()
    with io.open(p, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def channel_files(example_dir):
    """The files CHANNEL.md names, with their checksums and roles."""
    text = io.open(os.path.join(example_dir, 'CHANNEL.md'), encoding='utf-8').read()
    sums = dict(re.findall(r'^([0-9a-f]{64})  (\S+)$', text, re.M)[::1] and
                [(m[1], m[0]) for m in re.findall(r'^([0-9a-f]{64})  (\S+)$', text, re.M)])
    roles = {m[0]: m[1] for m in re.findall(r'^\| `(\S+)` \| (THRU|FEXT|NEXT) \|', text, re.M)}
    if not sums or not roles:
        die('could not read the file list out of CHANNEL.md')
    return sums, roles


def check_channels(channels, sums, roles, want_crosstalk):
    """Locate and verify the files this condition needs."""
    need = [n for n, r in roles.items() if want_crosstalk or r == 'THRU']
    found, missing, wrong = {}, [], []
    for name in need:
        hits = glob.glob(os.path.join(channels, '**', name), recursive=True)
        if not hits:
            missing.append(name)
            continue
        if sha256(hits[0]) != sums[name]:
            wrong.append(name)
        found[name] = hits[0]
    if missing:
        die('not found under %s:\n  %s\nSee %s/CHANNEL.md for where to get them.'
            % (channels, '\n  '.join(missing), EXAMPLE))
    if wrong:
        print('  WARNING: different content from the files the expected values came '
              'from:\n    %s' % '\n    '.join(wrong))
    thru = next(p for n, p in found.items() if roles[n] == 'THRU')
    fext = [p for n, p in found.items() if roles[n] == 'FEXT']
    next_ = [p for n, p in found.items() if roles[n] == 'NEXT']
    return thru, sorted(fext), sorted(next_)


def expected(example_dir, engine, condition):
    p = os.path.join(example_dir, 'expected_%s_%s.csv' % (engine, condition))
    with io.open(p, encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    return dict(zip(rows[0], rows[1]))


def report_row(result_dir):
    """The one data row of the COM report the engine wrote."""
    hits = sorted(glob.glob(os.path.join(result_dir, '**', '*results*.csv'), recursive=True))
    if not hits:
        die('the run wrote no results .csv under %s' % result_dir)
    with io.open(hits[-1], encoding='utf-8-sig') as fh:
        rows = [r for r in csv.reader(fh) if any(x.strip() for x in r)]
    # The report is written a column per line: name, then value.
    if len(rows) >= 2 and len(rows[0]) > 3:
        row = dict(zip(rows[0], rows[1]))
    else:
        row = {r[0].strip(): (r[1].strip() if len(r) > 1 else '') for r in rows}

    # Octave's CSV report keeps about six significant digits, which is not
    # enough to tell two implementations apart. The run also saves the result
    # struct, where every scalar is at full precision; prefer that.
    out_mat = os.path.join(result_dir, 'octave_result.mat')
    if os.path.isfile(out_mat):
        import numpy as np
        import scipy.io
        s = scipy.io.loadmat(out_mat, squeeze_me=True, struct_as_record=False)['r']
        for f in s._fieldnames:
            a = np.asarray(getattr(s, f))
            if a.dtype.kind in 'fiub' and a.size == 1:
                row[f] = repr(float(a.ravel()[0]))
    return row


def compare(got, want):
    """Rows of (key, expected, got, verdict), worst first."""
    out = []
    for k, tol in TOL.items():
        if k not in want:
            continue
        a, b = want.get(k, ''), got.get(k, '')
        try:
            fa, fb = float(a), float(b)
            d = abs(fa - fb)
            ok = d <= tol
            out.append((k, '%.12g' % fa, '%.12g' % fb, 'ok' if ok else 'DIFFERS by %.2e' % d))
        except (TypeError, ValueError):
            out.append((k, str(a), str(b), 'ok' if str(a) == str(b) else 'DIFFERS'))
    return out


def run_sicopr(cfg, thru, fext, next_, out):
    # The engine writes under its working directory, so the run happens in `out`.
    # --matlab-version matters: this example's workbook is a 4p16p0 one, and the
    # engine's default is 4p15p0, which reports two columns differently.
    cmd = [sys.executable, '-m', 'sicopr', os.path.abspath(cfg), os.path.abspath(thru)]
    if fext:
        cmd += ['--fext'] + [os.path.abspath(f) for f in fext]
    if next_:
        cmd += ['--next'] + [os.path.abspath(f) for f in next_]
    cmd += ['--matlab-version', '4p16p0']
    t0 = time.time()
    q = subprocess.run(cmd, cwd=out, capture_output=True, text=True, errors='replace')
    if q.returncode != 0:
        die('SiCoPR exited %d:\n%s' % (q.returncode, (q.stdout + q.stderr)[-1200:]))
    return time.time() - t0


def run_octave(cfg, thru, fext, next_, out, octave):
    from xlsx_to_com_mat import convert
    mat = os.path.join(out, 'config.mat')
    fwd = lambda p: os.path.abspath(p).replace('\\', '/')
    convert(cfg, mat, [('RESULT_DIR', fwd(out) + '/'), ('SAVE_FIGURES', '0'),
                       ('DISPLAY_WINDOW', '0'), ('CSV_REPORT', '1')])
    files = [thru] + list(fext) + list(next_)
    args = ', '.join(["'%s'" % fwd(mat), str(len(fext)), str(len(next_))]
                     + ["'%s'" % fwd(f) for f in files])
    ev = ("more off; addpath('%s'); r = com_ieee8023_4p16p0_octave_compat(%s); "
          "save('-v7', '%s', 'r'); exit(0);"
          % (fwd(os.path.join(ROOT, 'octave')), args,
             fwd(os.path.join(out, 'octave_result.mat'))))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    t0 = time.time()
    q = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       cwd=out, env=env, capture_output=True, text=True, errors='replace')
    if q.returncode != 0:
        die('Octave exited %d:\n%s' % (q.returncode, (q.stdout + q.stderr)[-1200:]))
    return time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--channels', required=True,
                    help='directory holding the unpacked channel files (searched recursively)')
    ap.add_argument('--example', default=EXAMPLE)
    ap.add_argument('--engine', choices=('sicopr', 'octave', 'both'), default='both')
    ap.add_argument('--condition', choices=('no_crosstalk', 'with_crosstalk', 'both'),
                    default='both')
    ap.add_argument('--octave', help='path to octave-cli (default: PATH)')
    ap.add_argument('--out', help='where results go (default: a temporary directory)')
    a = ap.parse_args(argv)

    example_dir = os.path.join(HERE, a.example)
    if not os.path.isdir(example_dir):
        die('no example directory %s' % example_dir)
    cfg = glob.glob(os.path.join(example_dir, '*.xlsx'))
    if not cfg:
        die('no configuration workbook in %s' % example_dir)
    cfg = cfg[0]
    sums, roles = channel_files(example_dir)

    octave = a.octave
    if a.engine in ('octave', 'both') and not octave:
        try:
            from xlsx_to_com_mat import find_octave
            octave = find_octave()
        except Exception:                                        # noqa: BLE001
            octave = None
    engines = ([('sicopr', None)] if a.engine == 'sicopr' else
               [('octave', octave)] if a.engine == 'octave' else
               [('sicopr', None), ('octave', octave)])
    conditions = ([a.condition] if a.condition != 'both'
                  else ['no_crosstalk', 'with_crosstalk'])
    out_root = a.out or tempfile.mkdtemp(prefix='sicopr_example_')
    print('example : %s' % a.example)
    print('config  : %s' % os.path.basename(cfg))
    print('results : %s\n' % out_root)

    bad = 0
    for cond in conditions:
        thru, fext, next_ = check_channels(a.channels, sums, roles,
                                           cond == 'with_crosstalk')
        for engine, tool in engines:
            if engine == 'octave' and not tool:
                print('%-7s %-14s SKIPPED: octave-cli not found (give --octave PATH)\n'
                      % (engine, cond))
                continue
            out = os.path.join(out_root, '%s_%s' % (engine, cond))
            os.makedirs(out, exist_ok=True)
            print('%-7s %-14s running %d file(s)...' % (engine, cond, 1 + len(fext) + len(next_)))
            secs = (run_sicopr(cfg, thru, fext, next_, out) if engine == 'sicopr'
                    else run_octave(cfg, thru, fext, next_, out, tool))
            rows = compare(report_row(out), expected(example_dir, engine, cond))
            worst = [r for r in rows if r[3] != 'ok']
            print('%-7s %-14s %.0f s   %s' % (
                engine, cond, secs,
                'matches the expected values' if not worst
                else '%d of %d differ' % (len(worst), len(rows))))
            for k, want, got, verdict in rows:
                flag = '' if verdict == 'ok' else '   <-- '
                print('        %-16s expected %-22s got %-22s%s%s'
                      % (k, want, got, flag, '' if verdict == 'ok' else verdict))
            bad += len(worst)
            print()

    print('all pinned values reproduced' if not bad
          else '%d value(s) differ; see above' % bad)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
