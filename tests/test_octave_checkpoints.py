"""SiCoPR against the Octave checkpoint goldens, every field of every stage.

The function tests see one function on synthetic inputs; the MATLAB stage
oracles see real cases but only the 35 numbers MATLAB reported. This sees the
whole pipeline on real cases: the reference, run under Octave with
COM_CHECKPOINT_DIR set, saved every struct it held at ten stage boundaries of
main (octave/patches/com_checkpoint.m), and SiCoPR, run here on the same
configuration and channels -- reading the inputs itself, not from the goldens
-- saves the same ten through com_ieee8023_'s _checkpoint(). Then every field
of every saved struct is compared.

Per field:
  * in Octave, absent in Python        FAIL, named
  * in Python, absent in Octave        reported, not failed (Python-only
                                       fields exist and are listed)
  * element count or 2-D layout differs      FAIL (shape)
  * same elements, 1-D against a MATLAB row or column   ORIENTATION, counted
                                       apart from value findings
  * values: scalars at rtol 1e-9; arrays at 1e-9 of the array's own peak;
    complex compared on real AND imaginary parts; non-finite values compared
    as patterns, because nan == nan is False
Everything is reported one row per field, failing rows first, and each
case/checkpoint pair is a check() so report.py sees it as a gate.

Goldens: $COM_OCTAVE_CHECKPOINTS/<version>/, written by
tools/regen_checkpoints.py. They are derived from IEEE channels that may not
be redistributed, so they live outside the repository and this test SKIPS,
loudly, where they are absent. When the case set moves to BeSS-generated
channels the goldens can ship under tests/golden/, and GOLDENS_SHIP below
flips the absence into a failure: that is the whole change.

Cases: COM_CHECKPOINT_CASES=all, or comma-separated ids. Unset, a quick subset
runs: the first without-crosstalk case for each package configuration.

    python tests/test_octave_checkpoints.py
    COM_CHECKPOINT_CASES=all python tests/test_octave_checkpoints.py

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import concurrent.futures as cf
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
from audit_check import check, finish                         # noqa: E402

# Flip to True when the goldens are generated from channels that may ship
# (BeSS) and committed under tests/golden/: absence then FAILS rather than
# skips. While they derive from IEEE channels they cannot ship, and a fresh
# public clone must stay green.
GOLDENS_SHIP = False

VERSION = os.environ.get('COM_CHECKPOINT_VERSION', '4p15p0')
RTOL = 1e-9
JOBS = int(os.environ.get('COM_CHECKPOINT_JOBS', '3'))
GOLD_ROOT = os.environ.get('COM_OCTAVE_CHECKPOINTS', '')
GOLD = os.path.join(GOLD_ROOT, VERSION) if GOLD_ROOT else ''
REPORT = os.environ.get('COM_CHECKPOINT_REPORT') or os.path.join(
    tempfile.gettempdir(), 'sicopr_checkpoints', VERSION)


# ----------------------------------------------------------- normalisation
def _is_matstruct(x):
    return hasattr(x, '_fieldnames')


def from_octave(x):
    """A loadmat(squeeze_me=False) value -> dict / list / ndarray / str.

    Arrays keep their MATLAB 2-D shape, because orientation is a finding.
    """
    if _is_matstruct(x):
        return {k: from_octave(getattr(x, k)) for k in x._fieldnames}
    if isinstance(x, np.ndarray):
        if x.dtype.kind in 'US':
            return ''.join(str(s) for s in x.ravel())
        if x.dtype == object:
            items = [from_octave(v) for v in x.ravel(order='F')]
            if items and all(isinstance(i, dict) for i in items):
                return items[0] if len(items) == 1 else items
            return items[0] if len(items) == 1 else items
        return x
    return x


def from_python(x, depth=0):
    if depth > 40:
        return None
    if isinstance(x, SimpleNamespace):
        return {k: from_python(v, depth + 1) for k, v in vars(x).items()}
    if isinstance(x, dict):
        return {str(k): from_python(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if x and all(isinstance(v, (int, float, complex, np.number, bool))
                     for v in x):
            return np.asarray(x)
        return [from_python(v, depth + 1) for v in x]
    if isinstance(x, (str, np.ndarray)) or x is None:
        return x
    if isinstance(x, (bool, int, float, complex, np.number, np.bool_)):
        return np.asarray(x)
    if hasattr(x, '__dict__'):
        return {k: from_python(v, depth + 1) for k, v in vars(x).items()
                if not k.startswith('_')}
    return x


# ------------------------------------------------------------- comparison
# Indices: MATLAB holds them 1-based, the port 0-based (AUDIT FINDING B16-D20,
# the convention OptFom_Find_Sample_Point, get_PSDs and OptFom_Compute_DFE
# share). Compared as octave == python + 1, element by element -- an explicit
# offset, never a tolerance. Each entry was confirmed exactly +1 on a real
# case before it went in: t_s 34151/34150 and DFE_taps_i 34183/34182 on
# wXtalk_T2_R12, iphase +1 on all five aggressors.
INDEX_BASE = {'fom_result.t_s', 'fom_result.DFE_taps_i',
              'fom_result.PSD_results.iphase'}

# Not computed by COM: where the run happened, the four output options
# tools/regen_checkpoints.py sets for the Octave run (RESULT_DIR, SAVE_FIGURES,
# DISPLAY_WINDOW, CSV_REPORT -- the first full run showed SAVE_FIGURES differing
# on the 14 cases whose workbook sets it), and a flag the Octave compat patch
# itself adds (COM_CommandLine_Parse sets OP.OCTAVE). Listed, not compared.
ENVIRONMENT = {'OP.RESULT_DIR', 'OP.SAVE_FIGURES', 'OP.DISPLAY_WINDOW',
               'OP.CSV_REPORT', 'output_args.config_file', 'OP.OCTAVE'}


# Present on both sides, but the value names the file that ran: ML 63 reads it
# off the reference's own file name, so Octave reports '4p15p0_octave_compat'
# and SiCoPR the release it emulates, '4p15p0'. Checked for presence only.
PRESENCE_ONLY = {'output_args.code_revision'}

# Compared twice: against the array's peak like every array, AND element by
# element relative to each element. Peak scaling cannot see the far tail, and
# the far tail is where DER_DFE and DER_MLSE are read: an FFT convolution in
# conv_fct left the noise CDF 1.3e-4 wrong at 1e-12 and 2.1e-16 where Octave
# had 5.2e-221, and every one of these rows passed at 1e-9 of the peak.
TAIL_RELATIVE = {'PDF.y', 'CDF'}
TAIL_FLOOR = np.finfo(float).tiny          # below this, subnormal: not compared

# DER is read off the noise CDF at -A_s (CDF_ev: find(PDF.x >= -A_s, 1)), and
# the reference sets the bin size to A_s/1000 (ML 526, param.delta_y), which
# puts -A_s ON a bin edge by construction. Which side of it the first bin falls
# is decided by the rounding of Min*BinSize + k*BinSize against A_s, so an A_s
# that differs in the last few bits -- the FFT residual in the pulse -- can
# move DER_DFE a whole bin (2-5 percent). MLSE convolves PDFs built on the
# same A_s-derived bins and reads its own tails, so DER_MLSE can jump too,
# with or without DER_DFE (4 of the 28 cases: DER_DFE agrees, DER_MLSE is off
# by 1.5-3.8 percent). A defect in the reference (upstream), not the port.
#
# A failing DER row becomes EDGE only when the case itself shows the whole
# difference is carried by an input the harness already accepts:
#   (a) the engine's own MLSE_U1_c_178A, fed the REFERENCE's inputs,
#       reproduces the reference's DER_DFE and DER_MLSE to RTOL -- the
#       function is faithful;
#   (b) swapping the fewest of the port's inputs (A_s, PDF, CDF: singly,
#       then in pairs, then all three) into the reference's set reproduces
#       the port's value of that field to RTOL -- nothing else contributes;
#   (c) each swapped input agrees with the reference's to RTOL, element by
#       element.
# Anything short of that stays FAIL.
DER_EDGE = {'COM_SNR_Struct.DER_DFE', 'COM_SNR_Struct.DER_MLSE',
            'output_args.DER_DFE', 'output_args.DER_MLSE'}


def _canon(path):
    """'chdata[3].TDR11' -> 'chdata.TDR11', for matching the sets above."""
    return re.sub(r'\[\d+\]', '', path)


def _empty(v):
    """MATLAB's [] as loadmat hands it back, or a struct of nothing but []."""
    if isinstance(v, np.ndarray):
        return v.size == 0
    if isinstance(v, dict):
        return all(_empty(x) for x in v.values())
    if isinstance(v, list):
        return len(v) == 0
    return v is None


def _numeric(a):
    return isinstance(a, np.ndarray) and a.dtype.kind in 'biufc'


def _compare_arrays(o, p, tail=False):
    """-> (status, detail, max_abs, max_rel, argmax) for two numeric arrays.

    tail: also compare element by element, relative to each element.
    """
    o_n, p_n = o.size, p.size
    if o_n == 0 and p_n == 0:
        return 'pass', 'both empty', 0.0, 0.0, ''
    if o_n != p_n:
        return 'FAIL', 'shape %s vs %s' % (o.shape, p.shape), '', '', ''
    orient = ''
    if o.shape != p.shape and not (o_n == 1):
        o2 = [d for d in o.shape if d != 1]
        p2 = [d for d in p.shape if d != 1]
        if o2 == p2:
            orient = 'orientation %s vs %s' % (o.shape, p.shape)
        else:
            return 'FAIL', 'layout %s vs %s' % (o.shape, p.shape), '', '', ''
    # MATLAB is column-major; a same-shaped numpy array is compared in place,
    # an orientation-only difference by the order the elements run in
    if o.shape == p.shape:
        ov, pv = o.ravel(order='F'), p.ravel(order='F')
    else:
        ov, pv = o.ravel(order='F'), p.ravel()
    ov = ov.astype(complex if (ov.dtype.kind == 'c' or pv.dtype.kind == 'c')
                   else float)
    pv = pv.astype(ov.dtype)
    fo, fp = np.isfinite(ov), np.isfinite(pv)
    if not np.array_equal(fo, fp):
        i = int(np.argmax(fo != fp))
        return 'FAIL', 'non-finite pattern differs at %d: %r vs %r' % (
            i, ov[i], pv[i]), '', '', i
    nf = ~fo
    if nf.any():
        # +inf, -inf and nan must match in kind, element by element
        def kind(v):
            return np.where(np.isnan(v.real) | np.isnan(getattr(v, 'imag', 0 * v.real)), 0,
                            np.sign(np.nan_to_num(v.real, nan=0, posinf=1, neginf=-1)))
        if not np.array_equal(kind(ov[nf]), kind(pv[nf])):
            return 'FAIL', 'non-finite kinds differ', '', '', ''
    if not fo.any():
        return ('ORIENT' if orient else 'pass'), orient or 'all non-finite', 0.0, 0.0, ''
    d = np.abs(ov[fo] - pv[fo])
    peak = float(np.max(np.abs(ov[fo])))
    scale = peak if peak > 0 else 1.0
    i = int(np.argmax(d))
    mabs, mrel = float(d[i]), float(d[i]) / scale
    idx = int(np.flatnonzero(fo)[i])
    if mrel > RTOL:
        return 'FAIL', 'value', mabs, mrel, idx
    if tail:
        big = fo & (np.abs(ov) >= TAIL_FLOOR)
        if big.any():
            r = np.abs(ov[big] - pv[big]) / np.abs(ov[big])
            j = int(np.argmax(r))
            if r[j] > RTOL:
                k = int(np.flatnonzero(big)[j])
                return ('FAIL', 'tail: element rel %.2e at %d (octave %.6g, '
                        'python %.6g)' % (r[j], k, abs(ov[k]), abs(pv[k])),
                        float(abs(ov[k] - pv[k])), float(r[j]), k)
            mrel = max(mrel, float(r[j]))
    return ('ORIENT' if orient else 'pass'), orient, mabs, mrel, idx


def compare(o, p, path, rows):
    """Walk the Octave tree; one row per leaf, and per missing field."""
    canon = _canon(path)
    if canon in ENVIRONMENT:
        rows.append((path, 'ENV', 'run environment, not compared',
                     '', '', '', ''))
        return
    if canon in PRESENCE_ONLY:
        rows.append((path, 'pass' if p is not None else 'FAIL',
                     'present; value names the file that ran' if p is not None
                     else 'absent in Python', '', '', '', ''))
        return
    if canon in INDEX_BASE and _numeric(np.asarray(o)) and p is not None:
        o = np.asarray(o, dtype=float) - 1.0      # to the port's 0-based form
        path = path + ' (MATLAB index - 1)'
    if isinstance(o, dict):
        # a 1x1 MATLAB struct and the port's one-element list of it
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], dict):
            p = p[0]
        # a struct array element MATLAB created with every field empty, where
        # the port holds nothing
        if p is None and _empty(o):
            rows.append((path, 'pass', 'empty struct / None', '', '', '', ''))
            return
        if not isinstance(p, dict):
            rows.append((path, 'FAIL', 'struct in Octave, %s in Python'
                         % type(p).__name__, '', '', '', ''))
            return
        for k in o:
            if k not in p:
                # MATLAB grows a field onto EVERY element of a struct array the
                # moment it is set on one, filled with []. The port's list of
                # namespaces does not, and both mean "no value here".
                if _empty(o[k]):
                    rows.append((path + '.' + k, 'pass',
                                 'MATLAB [] / absent in Python', '', '', '', ''))
                elif _canon(path + '.' + k) in ENVIRONMENT:
                    rows.append((path + '.' + k, 'ENV',
                                 'run environment, not compared', '', '', '', ''))
                else:
                    rows.append((path + '.' + k, 'FAIL', 'absent in Python',
                                 '', '', '', ''))
            else:
                compare(o[k], p[k], path + '.' + k, rows)
        for k in p:
            if k not in o:
                rows.append((path + '.' + k, 'PYONLY', 'absent in Octave',
                             '', '', '', ''))
        return
    if isinstance(o, list):
        if not isinstance(p, list) or len(p) != len(o):
            rows.append((path, 'FAIL', 'Octave list of %d, Python %s'
                         % (len(o), (len(p) if isinstance(p, list)
                                     else type(p).__name__)), '', '', '', ''))
            return
        for i, (a, b) in enumerate(zip(o, p)):
            compare(a, b, '%s[%d]' % (path, i), rows)
        return
    if isinstance(o, str):
        ok = isinstance(p, str) and p == o
        rows.append((path, 'pass' if ok else 'FAIL',
                     '' if ok else 'string %r vs %r' % (o[:40], str(p)[:40]),
                     '', '', '', ''))
        return
    if _numeric(o):
        if p is None:
            st = 'pass' if o.size == 0 else 'FAIL'
            rows.append((path, st, '' if st == 'pass' else 'None in Python',
                         '', '', '', str(o.shape)))
            return
        if isinstance(p, list) and not p and o.size == 0:
            rows.append((path, 'pass', 'both empty', '', '', '', ''))
            return
        pa = np.asarray(p) if not isinstance(p, np.ndarray) else p
        if not _numeric(pa):
            rows.append((path, 'FAIL', 'numeric in Octave, %s in Python'
                         % type(p).__name__, '', '', '', str(o.shape)))
            return
        st, det, mabs, mrel, idx = _compare_arrays(
            o, pa, tail=canon in TAIL_RELATIVE)
        rows.append((path, st, det, mabs, mrel, idx,
                     '%s|%s' % (o.shape, pa.shape)))
        return
    rows.append((path, 'SKIP', 'not comparable: %s' % type(o).__name__,
                 '', '', '', ''))


# --------------------------------------------------------------- running
def _load_octave(path):
    import scipy.io
    m = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False,
                         chars_as_strings=True)
    return {k: from_octave(v) for k, v in m.items() if not k.startswith('__')}


def _load_python(path):
    with open(path, 'rb') as fh:
        blob = pickle.load(fh)
    out = {}
    for k, raw in blob['values'].items():
        out[k] = from_python(pickle.loads(raw))
    return out, blob.get('failed', {})


def run_sicopr(case, outdir):
    os.makedirs(outdir, exist_ok=True)
    ck = os.path.join(outdir, 'checkpoints')
    # COM_CHECKPOINT_REUSE=1 compares the SiCoPR checkpoints already on disk
    # instead of running the engine again: for working on the comparison
    # itself. Never set it when the question is whether the engine changed.
    if os.environ.get('COM_CHECKPOINT_REUSE') == '1' and os.path.isdir(ck) \
            and any(f.endswith('.pkl') for f in os.listdir(ck)):
        return 'reused', ck
    os.makedirs(ck, exist_ok=True)
    for f in os.listdir(ck):
        os.remove(os.path.join(ck, f))
    cmd = [sys.executable, '-m', 'sicopr', case['config'], case['thru']]
    if case.get('fext'):
        cmd += ['--fext'] + list(case['fext'])
    if case.get('next'):
        cmd += ['--next'] + list(case['next'])
    cmd += ['--matlab-version', VERSION]
    env = dict(os.environ, COM_CHECKPOINT_DIR=os.path.abspath(ck))
    p = subprocess.run(cmd, cwd=outdir, env=env, capture_output=True,
                       text=True, errors='replace')
    with open(os.path.join(outdir, 'sicopr.log'), 'w', encoding='utf-8',
              errors='replace') as fh:
        fh.write('%s\nexit %s\n%s\n%s\n' % (' '.join(cmd), p.returncode,
                                           p.stdout, p.stderr))
    return p.returncode, ck


def _lookup(tree, path):
    """tree['a']['b'][2]['c'] for 'a.b[2].c'; a sentinel when absent."""
    cur = tree
    for tok in re.findall(r'[^.\[\]]+|\[\d+\]', path):
        if tok.startswith('['):
            i = int(tok[1:-1])
            if not isinstance(cur, list) or i >= len(cur):
                return _MISSING
            cur = cur[i]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return _MISSING
            cur = cur[tok]
    return cur


_MISSING = object()


def _mark_late(per_stage, stages, ck):
    """A field SiCoPR has not set YET is an ordering difference, not a missing
    value: re-class a stage's 'absent in Python' as LATE when a later SiCoPR
    checkpoint holds it. Reported apart; it does not fail the gate."""
    cache = {}

    def py(st):
        if st not in cache:
            pk = os.path.join(ck, st + '.pkl')
            cache[st] = _load_python(pk)[0] if os.path.isfile(pk) else {}
        return cache[st]

    for i, st in enumerate(stages):
        rows = per_stage[st]
        for j, r in enumerate(rows):
            if r[1] != 'FAIL' or r[2] != 'absent in Python':
                continue
            for later in stages[i + 1:]:
                if _lookup(py(later), r[0]) is not _MISSING:
                    rows[j] = (r[0], 'LATE', 'set later in SiCoPR, by %s'
                               % later) + tuple(r[3:])
                    break


def _ns_octave(path):
    """A golden .mat as the port's own types: structs as namespaces."""
    import scipy.io
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    def ns(x):
        if hasattr(x, '_fieldnames'):
            return SimpleNamespace(**{k: ns(getattr(x, k)) for k in x._fieldnames})
        return x
    return {k: ns(v) for k, v in m.items() if not k.startswith('__')}


def _raw_python(path):
    with open(path, 'rb') as fh:
        blob = pickle.load(fh)
    return {k: pickle.loads(v) for k, v in blob['values'].items()}


_ENGINE = {}


def _engine_module():
    """The sicopr.py the engine run used, not whichever this process imports.

    run_sicopr runs `python -m sicopr` from the report directory, which finds
    sicopr.py on PYTHONPATH first and the installed repository after. This
    process put _ROOT at the head of sys.path, so a bare `import sicopr` here
    would prove the DER edge with a DIFFERENT engine from the one under test --
    and a mutated MLSE would then hide behind the clean one.
    """
    if 'mod' not in _ENGINE:
        import importlib.util
        path = os.path.join(_ROOT, 'sicopr.py')
        for p in os.environ.get('PYTHONPATH', '').split(os.pathsep):
            if p and os.path.isfile(os.path.join(p, 'sicopr.py')):
                path = os.path.join(p, 'sicopr.py')
                break
        spec = importlib.util.spec_from_file_location('_engine_under_test', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ENGINE['mod'] = mod
    return _ENGINE['mod']


def _der_edge_proof(st10, stages, gold, ck):
    """-> {field suffix: (ok, detail)} for DER_DFE and DER_MLSE of st10."""
    tag = re.search(r'_pc\d+', st10).group(0)
    mine = [s for s in stages if tag + '_' in s + '_']

    def one(prefix):
        got = [s for s in mine if s.startswith(prefix)]
        return got[-1] if got else None     # 07: the second get_PSDs call
    s01, s05, s07, s08 = one('01_'), one('05_'), one('07_'), one('08_')
    if None in (s01, s05, s07, s08):
        why = 'DER edge: a checkpoint MLSE needs is missing'
        return {'DER_DFE': (False, why), 'DER_MLSE': (False, why)}
    g = lambda s: _ns_octave(os.path.join(gold, s + '.mat'))
    y = lambda s: _raw_python(os.path.join(ck, s + '.pkl'))
    o10, p10 = g(st10)['COM_SNR_Struct'], y(st10)['COM_SNR_Struct']
    o8, p8 = g(s08), y(s08)
    vec = lambda v: np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    ref = {'A_s': float(o10.A_s), 'PDF': o8['PDF'], 'CDF': vec(o8['CDF'])}
    port = {'A_s': float(p10.A_s), 'PDF': p8['PDF'], 'CDF': vec(p8['CDF'])}

    def rel(x, r):
        x, r = vec(x), vec(r)
        if x.shape != r.shape:
            return float('inf')
        big = np.abs(r) >= TAIL_FLOOR
        d = np.abs(x - r)
        return float(np.max(d[big] / np.abs(r[big]))) if big.any() else 0.0

    agree = {'A_s': rel(port['A_s'], ref['A_s']),
             'PDF': max(rel(port['PDF'].y, ref['PDF'].y),
                        rel(port['PDF'].x, ref['PDF'].x)),
             'CDF': rel(port['CDF'], ref['CDF'])}
    sicopr = _engine_module()
    param = g(s01)['param']
    blim = vec(g(s05)['fom_result'].MMSE_results.blim)
    A_ni = float(o10.A_ni)
    psd = g(s07)['PSD_results']

    def mlse(inp):
        M = sicopr.MLSE_U1_c_178A(param, blim, inp['A_s'], A_ni, inp['PDF'],
                                  inp['CDF'].copy(), psd)
        return {'DER_DFE': float(M.DER_DFE), 'DER_MLSE': float(M.DER_MLSE)}

    base = mlse(ref)
    import itertools
    combos = [c for n in (1, 2, 3) for c in itertools.combinations(sorted(ref), n)]
    swapped = {}

    def swap(c):
        if c not in swapped:
            swapped[c] = mlse(dict(ref, **{k: port[k] for k in c}))
        return swapped[c]
    x_o = vec(ref['PDF'].x)
    x_p = vec(port['PDF'].x)
    bins = (int(np.argmax(x_o >= -ref['A_s'])), int(np.argmax(x_p >= -port['A_s'])))
    out = {}
    for f in ('DER_DFE', 'DER_MLSE'):
        faithful = rel(base[f], getattr(o10, f)) <= RTOL
        carriers = [c for c in combos
                    if all(agree[k] <= RTOL for k in c)
                    and rel(swap(c)[f], getattr(p10, f)) <= RTOL][:1]
        detail = ('A_s bin edge (upstream): engine MLSE on the octave inputs '
                  'reproduces octave %s to %.1e; ' % (f, rel(base[f], getattr(o10, f))))
        if carriers:
            c = carriers[0]
            detail += ('the python %s alone (agree with octave to %.1e) '
                       'reproduce python %s to %.1e'
                       % ('+'.join(c), max(agree[k] for k in c), f,
                          rel(swap(c)[f], getattr(p10, f))))
        else:
            detail += ('no combination of inputs carries the difference (A_s %.1e, '
                       'PDF %.1e, CDF %.1e)' % (agree['A_s'], agree['PDF'],
                                                agree['CDF']))
        detail += '; -A_s first bin %d octave, %d python' % bins
        out[f] = (faithful and bool(carriers), detail)
    return out


def _mark_der_edge(per_stage, stages, gold, ck):
    for st in stages:
        rows = per_stage[st]
        hits = [j for j, r in enumerate(rows) if r[0] in DER_EDGE
                and r[1] == 'FAIL' and r[2] == 'value']
        if not hits:
            continue
        try:
            proof = _der_edge_proof(st, stages, gold, ck)
        except Exception as e:                  # the proof failing is a FAIL
            why = 'DER edge proof raised %s: %s' % (type(e).__name__, e)
            proof = {'DER_DFE': (False, why), 'DER_MLSE': (False, why)}
        for j in hits:
            r = rows[j]
            ok, why = proof['DER_DFE' if r[0].endswith('DER_DFE') else 'DER_MLSE']
            rows[j] = (r[0], 'EDGE' if ok else 'FAIL',
                       why if ok else 'value; ' + why) + tuple(r[3:])


def compare_case(cid, meta):
    gold = os.path.join(GOLD, cid)
    out = os.path.join(REPORT, cid)
    rc, ck = run_sicopr(meta, out)
    table = []
    stages = sorted(f[:-4] for f in os.listdir(gold)
                    if f.endswith('.mat') and f != 'results.mat')
    per_stage = {}
    for st in stages:
        rows = []
        pk = os.path.join(ck, st + '.pkl')
        if not os.path.isfile(pk):
            rows.append(('<stage>', 'FAIL', 'SiCoPR did not reach this stage '
                         '(exit %s)' % rc, '', '', '', ''))
        else:
            o = _load_octave(os.path.join(gold, st + '.mat'))
            p, failed = _load_python(pk)
            for k, why in failed.items():
                rows.append((k, 'FAIL', 'not pickled: %s' % why, '', '', '', ''))
            for k in o:
                if k not in p:
                    rows.append((k, 'FAIL', 'absent in Python', '', '', '', ''))
                else:
                    compare(o[k], p[k], k, rows)
        per_stage[st] = rows
    _mark_late(per_stage, stages, ck)
    _mark_der_edge(per_stage, stages, gold, ck)
    for st in stages:
        table += [(st,) + r for r in per_stage[st]]
    extra = sorted(f[:-4] for f in os.listdir(ck) if f.endswith('.pkl')
                   and f[:-4] not in stages)
    order = {'FAIL': 0, 'EDGE': 1, 'LATE': 2, 'ORIENT': 3, 'PYONLY': 4,
             'ENV': 5, 'SKIP': 6, 'pass': 7}
    table.sort(key=lambda r: (order.get(r[2], 5), r[0], r[1]))
    with open(os.path.join(out, 'fields.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['checkpoint', 'field', 'status', 'detail', 'max_abs',
                    'max_rel', 'index_of_max', 'octave|python shape'])
        w.writerows(table)
    return cid, rc, per_stage, extra


def crosscheck_matlab(cid, oracle):
    """Octave's results against MATLAB's own, where the case is in both.

    The goldens are only as good as Octave's fidelity to MATLAB. Where a
    checkpoint case is also one of the 208 MATLAB reference cases, every
    quantity MATLAB reported is compared with what Octave returned. Agreement
    is the evidence that the instrument is sound; a disagreement is either a
    known Octave divergence or a finding about the compat file, and both
    matter before a checkpoint failure is blamed on SiCoPR.
    -> list of (quantity, status, octave, matlab, rel)
    """
    case = oracle.get(cid)
    res = os.path.join(GOLD, cid, 'results.mat')
    if case is None or not os.path.isfile(res):
        return None
    import scipy.io
    r = scipy.io.loadmat(res, squeeze_me=True, struct_as_record=False)['r']
    rows = []
    items = list(case.get('values', {}).items()) + \
        list(case.get('vectors', {}).items())
    for q, mv in items:
        ov = getattr(r, q, None)
        if ov is None:
            rows.append((q, 'FAIL', 'absent in Octave results', mv, ''))
            continue
        try:
            # column-major: the result workbooks list a matrix field such as
            # C_diepad (2x3) as C_diepad_1..6 in MATLAB's (:) order. Flattening
            # Octave's copy row by row put three agreeing quantities out of
            # step with the workbook on every case in the first full run.
            o = np.atleast_1d(np.asarray(ov, dtype=float)).ravel(order='F')
            m = np.atleast_1d(np.asarray(mv, dtype=float)).ravel()
        except (TypeError, ValueError):
            rows.append((q, 'SKIP', 'not numeric', mv, ''))
            continue
        n = min(o.size, m.size)
        if n == 0:
            rows.append((q, 'SKIP', 'empty', mv, ''))
            continue
        o, m = o[:n], m[:n]
        fo, fm = np.isfinite(o), np.isfinite(m)
        if not np.array_equal(fo, fm):
            rows.append((q, 'FAIL', o.tolist(), m.tolist(), 'non-finite'))
            continue
        if not fo.any():
            rows.append((q, 'pass', o.tolist(), m.tolist(), 0.0))
            continue
        peak = float(np.max(np.abs(m[fm]))) or 1.0
        rel = float(np.max(np.abs(o[fo] - m[fm]))) / peak
        rows.append((q, 'pass' if rel <= RTOL else 'FAIL',
                     o.tolist() if n > 1 else float(o[0]),
                     m.tolist() if n > 1 else float(m[0]), rel))
    return rows


def _load_stage_oracle():
    path = os.environ.get('COM_STAGE_ORACLES', '')
    if not path or not os.path.isfile(path):
        return {}
    o = json.load(open(path, encoding='utf-8'))
    return {c['case_id']: c for c in o['cases']}


def main():
    man_path = os.path.join(GOLD, 'manifest.json') if GOLD else ''
    if not GOLD or not os.path.isfile(man_path):
        msg = ('octave checkpoints: %s. Generate them with\n'
               '    python tools/regen_checkpoints.py --cases CASES.json '
               '--version %s --out <dir>\n'
               'and point COM_OCTAVE_CHECKPOINTS at <dir>. They derive from '
               'the channels, so while those are IEEE public-area files they '
               'stay outside the repository.'
               % ('COM_OCTAVE_CHECKPOINTS is not set' if not GOLD_ROOT
                  else '%s not present' % man_path, VERSION))
        if GOLDENS_SHIP:
            check('octave_checkpoint_goldens_present', False, msg)
        else:
            print('SKIP ' + msg)
        return finish()

    man = json.load(open(man_path, encoding='utf-8'))
    cases = {k: v for k, v in man['cases'].items() if v.get('ok')}
    sel = os.environ.get('COM_CHECKPOINT_CASES', '').strip()
    if sel == 'all':
        chosen = sorted(cases)
    elif sel:
        chosen = [c for c in sel.split(',') if c in cases]
    else:
        seen, chosen = set(), []
        for cid in sorted(cases):
            m = re.match(r'woXtalk_T(\d)_', cid)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                chosen.append(cid)
    print('octave checkpoints %s: %d golden case(s), comparing %d%s'
          % (VERSION, len(cases), len(chosen),
             '' if sel else ' (set COM_CHECKPOINT_CASES=all for every case)'))
    print('per-field tables -> %s\n' % REPORT)

    with cf.ThreadPoolExecutor(max_workers=JOBS) as ex:
        results = list(ex.map(lambda c: compare_case(c, cases[c]), chosen))

    for cid, rc, per_stage, extra in results:
        print('%s  (SiCoPR exit %s)' % (cid, rc))
        for st, rows in per_stage.items():
            n = len(rows)
            nf = sum(1 for r in rows if r[1] == 'FAIL')
            nl = sum(1 for r in rows if r[1] == 'LATE')
            ne = sum(1 for r in rows if r[1] == 'EDGE')
            no = sum(1 for r in rows if r[1] == 'ORIENT')
            npy = sum(1 for r in rows if r[1] == 'PYONLY')
            first = next((r for r in rows if r[1] == 'FAIL'), None)
            worst = max((r[4] for r in rows if isinstance(r[4], float)
                         and r[1] not in ('FAIL', 'EDGE')), default=0.0)
            print('   %-28s fields %5d  fail %4d  late %3d  orient %4d  '
                  'py-only %4d%s  worst passing rel %.1e%s'
                  % (st, n, nf, nl, no, npy,
                     ('  EDGE %d' % ne) if ne else '', worst,
                     ('  first: %s (%s)' % (first[0], first[2][:50]))
                     if first else ''))
            check('%s__%s_matches_octave' % (cid, st), nf == 0,
                  '%d field(s) fail; first %s: %s'
                  % (nf, first[0], first[2]) if first else '')
        if extra:
            print('   SiCoPR-only stages: %s' % ', '.join(extra))

    oracle = _load_stage_oracle()
    if not oracle:
        print('\nOctave-vs-MATLAB cross-check SKIPPED: COM_STAGE_ORACLES is not '
              'set. Point it at matlab_stage_oracles.json to check the goldens '
              'themselves against MATLAB.')
    else:
        print('\nOctave against MATLAB (is the instrument sound?)')
        for cid in chosen:
            rows = crosscheck_matlab(cid, oracle)
            if rows is None:
                print('   %-18s not a MATLAB reference case' % cid)
                continue
            bad = [r for r in rows if r[1] == 'FAIL']
            worst = max((r[4] for r in rows if isinstance(r[4], float)),
                        default=0.0)
            print('   %-18s %2d quantities, %d disagree, worst rel %.2e%s'
                  % (cid, len(rows), len(bad), worst,
                     ('  e.g. %s: octave %s matlab %s'
                      % (bad[0][0], str(bad[0][2])[:30], str(bad[0][3])[:30]))
                     if bad else ''))
            with open(os.path.join(REPORT, cid, 'octave_vs_matlab.csv'), 'w',
                      newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(['quantity', 'status', 'octave', 'matlab', 'rel'])
                w.writerows(sorted(rows, key=lambda r: (r[1] != 'FAIL', r[0])))
            check('%s__octave_matches_matlab' % cid, not bad,
                  '; '.join('%s octave %s matlab %s' % (b[0], str(b[2])[:24],
                                                        str(b[3])[:24])
                            for b in bad[:4]))
    return finish()


if __name__ == '__main__':
    main()
