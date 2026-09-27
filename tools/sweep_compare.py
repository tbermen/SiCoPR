"""sweep_compare.py — Run the three EQ-search methods on one channel and capture
the per-candidate trajectory logs used by the sweep-vs-adaptive comparison plots.

The three methods are selected through the two optimize_fom switches:
    full_grid : Local Search = 0                 (exhaustive)
    legacy    : Local Search = N, NonZeroLSMethod = 0   (OptFom_Local_Search)
    adaptive  : Local Search = N, NonZeroLSMethod = 1   (OptFom_Adaptive_Local_Search)

For each run the opt-in logger in optimize_fom (sicopr.SWEEP_LOG_CSV) writes one row
per TX-FFE candidate considered (method, EQ indices, tap vector, candidate FOM,
running best FOM, evaluated/skipped). The adaptive run additionally enables the
ALS radius log (sicopr.ALS_LOG_CSV) for the adaptive-radius diagnostic.

Outputs (in --out): full_grid_log.csv, legacy_log.csv, adaptive_log.csv,
adaptive_radius_log.csv, summary.json.

Usage:
    python sweep_compare.py <config.xlsx> <thru.s4p> [--fext a.s4p ...] [--next n.s4p ...]
                            [--local-search 2] [--max-ctle 6] [--out sweep_results]
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np

_TAP_RE = re.compile(r'^tx_ffe_c[mp]\d+_values$')


def _subsample(vals, n):
    """Evenly subsample a list to <= n values, preserving the endpoints/range."""
    vals = list(np.asarray(vals).ravel())
    if n is None or len(vals) <= n:
        return vals
    idx = np.unique(np.round(np.linspace(0, len(vals) - 1, n)).astype(int))
    return [vals[i] for i in idx]


def apply_grid_reduction(param, max_ctle=None, max_tap_vals=None):
    """Shrink the EQ search grid in place to keep the full sweep tractable.

    Shared with fom_com_probe.py: the EQ indices in the sweep logs are relative
    to the *reduced* grid, so the probe must reduce identically before pinning.
    """
    if max_ctle:
        param.ctle_gdc_values = np.asarray(param.ctle_gdc_values).ravel()[:max_ctle]
    if max_tap_vals:
        for fld in [f for f in vars(param) if _TAP_RE.match(f)]:
            setattr(param, fld, np.asarray(_subsample(getattr(param, fld), max_tap_vals)))
    return param

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import sicopr  # assembled COM module


def _extract_case(results):
    if isinstance(results, (list, tuple)):
        cases = [r for r in results if r is not None]
        return cases[0] if cases else None
    return results


METHOD_SWITCHES = {  # label -> (LOCAL_SEARCH uses local_search?, NonZeroLSMethod)
    'full_grid': (False, 0),
    'legacy': (True, 0),
    'adaptive': (True, 1),
}


def run_methods(config, thru, fext=(), next_=(), local_search=2,
                max_ctle=None, max_tap_vals=None, out_dir='sweep_results',
                methods=None):
    """Run the requested EQ-search methods on one channel.

    methods: subset of METHOD_SWITCHES keys, in run order. Defaults to all three.
    Dropping 'legacy' roughly halves corpus wall time; keep 'full_grid' or the
    dCOM / same_EQ comparison columns have no reference to compare against.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = [thru, *fext, *next_]
    labels = list(methods) if methods else ['full_grid', 'legacy', 'adaptive']
    bad = [m for m in labels if m not in METHOD_SWITCHES]
    if bad:
        raise ValueError(f'unknown method(s): {bad}; known: {sorted(METHOD_SWITCHES)}')
    methods = [(m, local_search if METHOD_SWITCHES[m][0] else 0, METHOD_SWITCHES[m][1])
               for m in labels]

    orig_read = sicopr.read_ParamConfigFile
    summary = {'config': os.path.basename(config), 'thru': os.path.basename(thru),
               'local_search': local_search, 'max_ctle': max_ctle,
               'max_tap_vals': max_tap_vals, 'methods': {}}
    try:
        for label, ls, nz in methods:
            def patched(cf, OP, _ls=ls, _nz=nz):
                param, OP = orig_read(cf, OP)
                param.LOCAL_SEARCH = _ls
                param.NonZeroLSMethod = _nz
                apply_grid_reduction(param, max_ctle, max_tap_vals)
                return param, OP

            sicopr.read_ParamConfigFile = patched
            sicopr.SWEEP_LOG_CSV = os.path.join(out_dir, f'{label}_log.csv')
            sicopr.SWEEP_METHOD_LABEL = label
            sicopr.ALS_LOG_CSV = (os.path.join(out_dir, 'adaptive_radius_log.csv')
                               if label == 'adaptive' else None)
            # fresh radius log each run (append_csv_row appends, so clear first)
            if sicopr.ALS_LOG_CSV and os.path.isfile(sicopr.ALS_LOG_CSV):
                os.remove(sicopr.ALS_LOG_CSV)

            print(f'\n=== {label}: Local Search={ls}, NonZeroLSMethod={nz} ===', flush=True)
            t0 = time.time()
            sicopr.COM_MATLAB_VERSION = '4p15p0'   # the 208-case reference is 4p15p0
            results = sicopr._run_com(config, len(fext), len(next_), files, export_mat=False)
            dt = time.time() - t0
            sicopr.SWEEP_LOG_CSV = None
            sicopr.ALS_LOG_CSV = None

            r = _extract_case(results)
            com_db = float(getattr(r, 'COM_dB', float('nan'))) if r is not None else float('nan')
            log_csv = os.path.join(out_dir, f'{label}_log.csv')
            st = _log_stats(log_csv)
            summary['methods'][label] = {
                'COM_dB': com_db, 'best_FOM_dB': st['best_FOM_dB'], 'wall_s': round(dt, 1),
                'n_evaluated': st['n_evaluated'], 'n_skipped': st['n_skipped'],
                'n_considered': st['n_evaluated'] + st['n_skipped'],
                'best_EQ': st['best_EQ'], 'log_csv': os.path.basename(log_csv)}
            print(f'  COM_dB={com_db:.4f}  best_FOM={st["best_FOM_dB"]:.3f}  '
                  f'evaluated={st["n_evaluated"]}  skipped={st["n_skipped"]}  '
                  f'wall={dt:.1f}s', flush=True)
            print(f'  best_EQ={st["best_EQ"]}', flush=True)
    finally:
        sicopr.read_ParamConfigFile = orig_read
        sicopr.SWEEP_LOG_CSV = None
        sicopr.ALS_LOG_CSV = None

    # speedups + grid coverage relative to full grid (based on candidates actually
    # evaluated — the real compute — not merely considered)
    fg = summary['methods'].get('full_grid', {})
    full_eval = fg.get('n_evaluated') or 1
    full_wall = fg.get('wall_s') or 1e-9
    for label, m in summary['methods'].items():
        m['pct_of_full_grid_evaluated'] = round(100.0 * m['n_evaluated'] / full_eval, 1)
        m['speedup_vs_full_grid'] = round(full_wall / max(m['wall_s'], 1e-9), 2)
        # Did this method land on the same EQ operating point as the full grid,
        # or a different one that merely scores alike? Equal FOM with unequal
        # EQ is the signature of a tie broken differently -- and COM can differ.
        if fg.get('best_EQ') is not None and m.get('best_EQ') is not None:
            m['same_EQ_as_full_grid'] = (m['best_EQ'] == fg['best_EQ'])
            m['dCOM_vs_full_grid'] = round(m['COM_dB'] - fg['COM_dB'], 6)

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote logs + summary.json to {out_dir}/')
    return summary


# The EQ indices plus the sampling phase that rode along with the winning FOM.
# best_itick/best_cursor_i are read from the row *after* the winning update, so
# they describe the operating point that COM was ultimately evaluated at.
EQ_KEYS = ('gffe_index', 'ctle_index', 'lp_index', 'txffe_index', 'tx_taps',
           'best_itick', 'best_cursor_i')


def _log_stats(csv_path):
    """Summarise a sweep log: counts, best FOM, and the *winning EQ setting*.

    The winner is the first evaluated candidate attaining the maximum
    candidate_FOM. That matches optimize_fom's strict `THIS.FOM > BEST.FOM`
    update (sicopr.py:12922), which keeps the earliest member of any tie.

    Capturing the EQ setting -- not just the score -- is what makes it possible
    to tell "the methods found the same operating point" from "the methods found
    different operating points that happen to tie on FOM". The latter is the
    case that can move COM.
    """
    out = {'n_evaluated': 0, 'n_skipped': 0, 'best_FOM_dB': float('nan'),
           'best_EQ': None}
    if not os.path.isfile(csv_path):
        return out

    best = float('-inf')
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('evaluated') != '1':
                out['n_skipped'] += 1
                continue
            out['n_evaluated'] += 1
            try:
                fom = float(row['candidate_FOM'])
            except (TypeError, ValueError):
                continue
            if fom > best:
                best = fom
                out['best_EQ'] = {k: row.get(k) for k in EQ_KEYS}

    if best != float('-inf'):
        out['best_FOM_dB'] = best
    return out


def main():
    ap = argparse.ArgumentParser(description='EQ-search method comparison (full grid / legacy / adaptive)')
    ap.add_argument('config')
    ap.add_argument('thru')
    ap.add_argument('--fext', nargs='*', default=[])
    ap.add_argument('--next', nargs='*', default=[], dest='next_')
    ap.add_argument('--local-search', type=int, default=2)
    ap.add_argument('--max-ctle', type=int, default=None,
                    help='cap the CTLE sweep to the first N values to keep the full grid tractable')
    ap.add_argument('--max-tap-vals', type=int, default=None,
                    help='evenly subsample each TX-FFE tap to <= N values (shrinks the grid)')
    ap.add_argument('--out', default='sweep_results')
    args = ap.parse_args()
    run_methods(args.config, args.thru, args.fext, args.next_,
                local_search=args.local_search, max_ctle=args.max_ctle,
                max_tap_vals=args.max_tap_vals, out_dir=args.out)


if __name__ == '__main__':
    main()
