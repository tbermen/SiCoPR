"""corpus_sweep.py — run the EQ-search method comparison across a corpus of channels.

Everything upstream of this measures one channel. A single channel can only ever say
"lossless here"; a standards contribution needs a *distribution*. This driver runs
sweep_compare (and optionally fom_com_probe) over N channels and aggregates:

  dCOM      per method vs full_grid -- how often, and by how much, pruning changes COM
  speedup   the benefit being claimed
  same_EQ   whether the pruned search found the same operating point at all
  regret    (probe only) COM given up by trusting the FOM ranking, which is paid by
            EVERY method including exhaustive full-grid
  flips     channels where a method crosses the pass/fail threshold differently

The tail is the point. A good mean dCOM proves nothing; one channel that flips a
3 dB pass into a fail sinks the proposal. Aggregates report max and P95, not just mean.

Cost is the binding constraint -- roughly an hour per channel for all three methods at
the reference grid, so a 7-channel corpus is most of a day. Mitigations:
  --methods full_grid,adaptive   drops ~40% of the wall time
  --max-ctle / --max-tap-vals    shrink the grid (must match across the corpus)
  --probe-top-k 0                skip the regret half
Runs are checkpointed per channel and skipped if already complete, so an interrupted
corpus resumes where it left off rather than starting over.

Usage:
    python corpus_sweep.py <config.xlsx> --channel-dir <channel-dir> [--crosstalk]
                           [--methods full_grid,legacy,adaptive] [--local-search 2]
                           [--max-ctle 3] [--max-tap-vals 3] [--probe-top-k 10]
                           [--threshold 3.0] [--out corpus_results] [--dry-run] [--force]

    python corpus_sweep.py --aggregate-only --out corpus_results
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # tools/
import sweep_compare
import fom_com_probe

# Rough per-method minutes per channel at --max-ctle 3 --max-tap-vals 3, measured
# 2026-08-06. Only used for the upfront estimate; scales with the grid knobs.
_EST_MIN = {'full_grid': 33.0, 'legacy': 24.0, 'adaptive': 2.5}
_EST_PROBE_MIN_PER_CAND = 0.37

RUN_FIELDS = ['channel', 'method', 'COM_dB', 'best_FOM_dB', 'n_evaluated', 'n_skipped',
              'pct_of_full_grid_evaluated', 'speedup_vs_full_grid', 'same_EQ_as_full_grid',
              'dCOM_vs_full_grid', 'wall_s', 'ctle_index', 'lp_index', 'txffe_index',
              'tx_taps', 'best_itick']
PROBE_FIELDS = ['channel', 'fom_rank', 'FOM_dB', 'COM_dB', 'dFOM_vs_rank1', 'dCOM_vs_rank1']


# -- channel discovery --------------------------------------------------------
def discover_channels(root, crosstalk=False):
    """Group s4p files into channels by the stem before _thru1 / _xtalkN_(Fext|Next)."""
    channels = []
    for thru in sorted(glob.glob(os.path.join(root, '*_thru1.s4p'))):
        stem = os.path.basename(thru)[:-len('_thru1.s4p')]
        ch = {'id': stem, 'thru': thru, 'fext': [], 'next': []}
        if crosstalk:
            for x in sorted(glob.glob(os.path.join(root, f'{stem}_xtalk*.s4p'))):
                if re.search(r'_Fext\.s4p$', x, re.I):
                    ch['fext'].append(x)
                elif re.search(r'_Next\.s4p$', x, re.I):
                    ch['next'].append(x)
        channels.append(ch)
    return channels


def _short(cid):
    """Compact label for tables: pull the distinguishing length out of the stem."""
    m = re.search(r'BPK_(\d+mm)_', cid)
    return m.group(1) if m else cid[:24]


# -- per-channel execution ----------------------------------------------------
def _is_complete(cdir, methods, probe_top_k):
    p = os.path.join(cdir, 'summary.json')
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            s = json.load(f)
    except (ValueError, OSError):
        return False
    if not all(m in s.get('methods', {}) for m in methods):
        return False
    if probe_top_k and not os.path.isfile(os.path.join(cdir, 'fom_com_probe.csv')):
        return False
    return True


def run_one(config, ch, out_dir, methods, local_search, max_ctle, max_tap_vals,
            probe_top_k, force):
    cdir = os.path.join(out_dir, ch['id'])
    if not force and _is_complete(cdir, methods, probe_top_k):
        print(f'  [skip] already complete: {ch["id"]}', flush=True)
        return 'skipped', None

    os.makedirs(cdir, exist_ok=True)
    t0 = time.time()
    sweep_compare.run_methods(
        config, ch['thru'], ch['fext'], ch['next'], local_search=local_search,
        max_ctle=max_ctle, max_tap_vals=max_tap_vals, out_dir=cdir, methods=methods)

    if probe_top_k:
        probe_src = 'full_grid' if 'full_grid' in methods else methods[0]
        fom_com_probe.probe(config, ch['thru'], ch['fext'], ch['next'],
                            sweep_dir=cdir, method=probe_src, top_k=probe_top_k)

    return 'ok', round(time.time() - t0, 1)


# -- aggregation --------------------------------------------------------------
def _load_channel(cdir):
    out = {'summary': None, 'probe': None}
    p = os.path.join(cdir, 'summary.json')
    if os.path.isfile(p):
        with open(p) as f:
            out['summary'] = json.load(f)
    p = os.path.join(cdir, 'fom_com_probe.csv')
    if os.path.isfile(p):
        with open(p, newline='') as f:
            out['probe'] = list(csv.DictReader(f))
    p = os.path.join(cdir, 'fom_com_probe_summary.json')
    if os.path.isfile(p):
        with open(p) as f:
            out['probe_summary'] = json.load(f)
    return out


def _stats(vals, dig=6):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {'n': int(v.size), 'mean': round(float(v.mean()), dig),
            'median': round(float(np.median(v)), dig),
            'p95': round(float(np.percentile(v, 95)), dig),
            'min': round(float(v.min()), dig), 'max': round(float(v.max()), dig)}


def aggregate(out_dir, threshold=3.0):
    cdirs = sorted(d for d in glob.glob(os.path.join(out_dir, '*'))
                   if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'summary.json')))
    runs, probes = [], []
    per_channel = {}

    for cdir in cdirs:
        cid = os.path.basename(cdir)
        data = _load_channel(cdir)
        s = data['summary']
        if not s:
            continue
        per_channel[cid] = data
        for label, m in s['methods'].items():
            eq = m.get('best_EQ') or {}
            runs.append({
                'channel': cid, 'method': label, 'COM_dB': m.get('COM_dB'),
                'best_FOM_dB': m.get('best_FOM_dB'), 'n_evaluated': m.get('n_evaluated'),
                'n_skipped': m.get('n_skipped'),
                'pct_of_full_grid_evaluated': m.get('pct_of_full_grid_evaluated'),
                'speedup_vs_full_grid': m.get('speedup_vs_full_grid'),
                'same_EQ_as_full_grid': m.get('same_EQ_as_full_grid'),
                'dCOM_vs_full_grid': m.get('dCOM_vs_full_grid'), 'wall_s': m.get('wall_s'),
                'ctle_index': eq.get('ctle_index'), 'lp_index': eq.get('lp_index'),
                'txffe_index': eq.get('txffe_index'), 'tx_taps': eq.get('tx_taps'),
                'best_itick': eq.get('best_itick')})
        for r in (data['probe'] or []):
            probes.append({'channel': cid, 'fom_rank': r['fom_rank'], 'FOM_dB': r['FOM_dB'],
                           'COM_dB': r['COM_dB'], 'dFOM_vs_rank1': r['dFOM_vs_rank1'],
                           'dCOM_vs_rank1': r['dCOM_vs_rank1']})

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        w.writerows(runs)
    if probes:
        with open(os.path.join(out_dir, 'probe.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=PROBE_FIELDS)
            w.writeheader()
            w.writerows(probes)

    # ---- distribution stats, per method, vs full_grid ----
    labels = []
    for r in runs:
        if r['method'] not in labels:
            labels.append(r['method'])

    summary = {'out_dir': out_dir, 'n_channels': len(per_channel),
               'threshold_dB': threshold, 'methods': {}}

    for label in labels:
        rows = [r for r in runs if r['method'] == label]
        d = [r['dCOM_vs_full_grid'] for r in rows]
        d_abs = [abs(x) for x in d if x is not None]
        same = [r['same_EQ_as_full_grid'] for r in rows if r['same_EQ_as_full_grid'] is not None]
        entry = {
            'n_channels': len(rows),
            'speedup': _stats([r['speedup_vs_full_grid'] for r in rows], 2),
            'pct_grid_evaluated': _stats([r['pct_of_full_grid_evaluated'] for r in rows], 1),
            'dCOM_vs_full_grid': _stats(d),
            'abs_dCOM_vs_full_grid': _stats(d_abs),
            'n_exact_zero_dCOM': sum(1 for x in d if x == 0.0),
            'n_same_EQ': sum(1 for x in same if x),
            'n_diff_EQ': sum(1 for x in same if not x)}

        # pass/fail flips: the tail risk that actually matters
        flips = []
        for r in rows:
            fg = next((x for x in runs
                       if x['channel'] == r['channel'] and x['method'] == 'full_grid'), None)
            if not fg or fg['COM_dB'] is None or r['COM_dB'] is None:
                continue
            if (fg['COM_dB'] >= threshold) != (r['COM_dB'] >= threshold):
                flips.append({'channel': r['channel'], 'full_grid_COM': fg['COM_dB'],
                              f'{label}_COM': r['COM_dB']})
        entry['n_passfail_flips'] = len(flips)
        entry['passfail_flips'] = flips
        summary['methods'][label] = entry

    # ---- regret, from the probe ----
    regrets, rhos = [], []
    for cid, data in per_channel.items():
        ps = data.get('probe_summary') or {}
        if ps.get('COM_regret_dB') is not None:
            regrets.append(ps['COM_regret_dB'])
        if ps.get('spearman_rho_FOM_vs_COM') is not None:
            rhos.append(ps['spearman_rho_FOM_vs_COM'])
    if regrets:
        summary['fom_proxy'] = {'n_channels': len(regrets),
                                'COM_regret_dB': _stats(regrets),
                                'spearman_rho_FOM_vs_COM': _stats(rhos, 4)}

    with open(os.path.join(out_dir, 'corpus_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary, runs


def report(summary, runs):
    print('\n' + '=' * 74)
    print(f'CORPUS: {summary["n_channels"]} channels  ->  {summary["out_dir"]}/')
    print('=' * 74)

    chans = []
    for r in runs:
        if r['channel'] not in chans:
            chans.append(r['channel'])
    labels = list(summary['methods'])

    print(f'\n{"channel":>10}  ' + '  '.join(f'{l:>22}' for l in labels))
    for c in chans:
        cells = []
        for l in labels:
            r = next((x for x in runs if x['channel'] == c and x['method'] == l), None)
            if r is None or r['COM_dB'] is None:
                cells.append(f'{"--":>22}')
            else:
                d = r['dCOM_vs_full_grid']
                ds = '' if d is None else (' =' if d == 0.0 else f' {d:+.4f}')
                cells.append(f'{r["COM_dB"]:>13.6f}{ds:>9}')
        print(f'{_short(c):>10}  ' + '  '.join(cells))
    print('\n  (COM in dB; second column is dCOM vs full_grid, "=" means bit-identical)')

    for label, m in summary['methods'].items():
        if label == 'full_grid':
            continue
        print(f'\n-- {label} --')
        sp, ad = m['speedup'], m['abs_dCOM_vs_full_grid']
        if sp:
            print(f'   speedup        median {sp["median"]}x   min {sp["min"]}x   max {sp["max"]}x')
        if ad:
            print(f'   |dCOM| (dB)    median {ad["median"]}   P95 {ad["p95"]}   MAX {ad["max"]}')
        print(f'   exact-zero dCOM: {m["n_exact_zero_dCOM"]}/{m["n_channels"]} channels')
        print(f'   same EQ point:   {m["n_same_EQ"]}/{m["n_same_EQ"] + m["n_diff_EQ"]} channels')
        flips = m['n_passfail_flips']
        mark = 'OK' if flips == 0 else f'!! {flips} FLIP(S) -- investigate'
        print(f'   pass/fail flips at {summary["threshold_dB"]} dB: {mark}')

    fp = summary.get('fom_proxy')
    if fp:
        rg, rh = fp['COM_regret_dB'], fp['spearman_rho_FOM_vs_COM']
        print(f'\n-- FOM as a proxy for COM ({fp["n_channels"]} channels probed) --')
        print(f'   COM regret (dB)  median {rg["median"]}   P95 {rg["p95"]}   MAX {rg["max"]}')
        print(f'   Spearman rho     median {rh["median"]}   min {rh["min"]}   max {rh["max"]}')
        print('   (regret is the cost of FOM-driven search itself -- every method pays it)')
    print('=' * 74)


# -- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Corpus-wide EQ-search method comparison')
    ap.add_argument('config', nargs='?', help='COM config .xlsx (not needed with --aggregate-only)')
    ap.add_argument('--channel-dir', help='directory of *_thru1.s4p channel files')
    ap.add_argument('--crosstalk', action='store_true',
                    help='include matching _xtalkN_Fext/_Next files (slower, more realistic)')
    ap.add_argument('--methods', default='full_grid,legacy,adaptive',
                    help='comma-separated subset; keep full_grid for the dCOM reference')
    ap.add_argument('--local-search', type=int, default=2)
    ap.add_argument('--max-ctle', type=int, default=3)
    ap.add_argument('--max-tap-vals', type=int, default=3)
    ap.add_argument('--probe-top-k', type=int, default=10,
                    help='per-channel FOM->COM probe depth; 0 disables (skips regret)')
    ap.add_argument('--threshold', type=float, default=3.0,
                    help='COM pass/fail threshold used for flip detection')
    ap.add_argument('--limit', type=int, default=None, help='only run the first N channels')
    ap.add_argument('--out', default='corpus_results')
    ap.add_argument('--dry-run', action='store_true', help='list channels + cost estimate, run nothing')
    ap.add_argument('--force', action='store_true', help='re-run channels already complete')
    ap.add_argument('--aggregate-only', action='store_true', help='re-aggregate existing results')
    args = ap.parse_args()

    if args.aggregate_only:
        summary, runs = aggregate(args.out, args.threshold)
        report(summary, runs)
        return

    if not args.config or not args.channel_dir:
        ap.error('config and --channel-dir are required unless --aggregate-only')

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    channels = discover_channels(args.channel_dir, args.crosstalk)
    if args.limit:
        channels = channels[:args.limit]
    if not channels:
        raise SystemExit(f'no *_thru1.s4p found under {args.channel_dir}')

    est = sum(_EST_MIN.get(m, 10.0) for m in methods)
    est += args.probe_top_k * _EST_PROBE_MIN_PER_CAND
    todo = [c for c in channels
            if args.force or not _is_complete(os.path.join(args.out, c['id']),
                                              methods, args.probe_top_k)]

    print(f'Corpus: {len(channels)} channel(s) under {args.channel_dir}')
    for c in channels:
        done = '' if c in todo else '  [complete]'
        xt = f'  (+{len(c["fext"])} FEXT, +{len(c["next"])} NEXT)' if args.crosstalk else ''
        print(f'  {_short(c["id"]):>8}  {c["id"]}{xt}{done}')
    print(f'\nMethods: {methods}   probe-top-k: {args.probe_top_k}')
    print(f'Rough estimate: ~{est:.0f} min/channel x {len(todo)} to run '
          f'= ~{est * len(todo) / 60:.1f} h  (scales with --max-ctle/--max-tap-vals)')
    if args.dry_run:
        print('\n--dry-run: nothing executed.')
        return
    if not todo:
        print('\nAll channels already complete; aggregating.')

    t0 = time.time()
    for i, ch in enumerate(channels, 1):
        print(f'\n{"=" * 74}\n[{i}/{len(channels)}] {ch["id"]}\n{"=" * 74}', flush=True)
        try:
            status, wall = run_one(args.config, ch, args.out, methods, args.local_search,
                                   args.max_ctle, args.max_tap_vals, args.probe_top_k,
                                   args.force)
            if status == 'ok':
                print(f'  [done] {ch["id"]} in {wall}s', flush=True)
        except Exception:
            # One bad channel must not cost the rest of a multi-hour corpus.
            print(f'  [FAIL] {ch["id"]}:\n{traceback.format_exc()}', file=sys.stderr, flush=True)
            with open(os.path.join(args.out, 'failures.log'), 'a') as f:
                f.write(f'--- {ch["id"]} ---\n{traceback.format_exc()}\n')

    print(f'\nCorpus wall time: {(time.time() - t0) / 60:.1f} min')
    summary, runs = aggregate(args.out, args.threshold)
    report(summary, runs)


if __name__ == '__main__':
    main()
