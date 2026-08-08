"""fom_com_probe.py — measure how well FOM predicts COM near the FOM optimum.

The EQ search maximises FOM, but the pass/fail metric is COM. COM is deliberately
*not* computed during the search (optimize_fom runs with OP.COMPUTE_COM = False,
com.py:448) — it is evaluated once at the end from the winning EQ setting, because
per-candidate COM would mean the full PDF/CDF convolution on every grid point.

That makes FOM a proxy, and any local-search heuristic that prunes in FOM space is
safe only to the extent that COM is flat across the top of the FOM surface. This
script measures that flatness directly:

  1. take the top-K candidates by FOM from a sweep log (full_grid by default),
  2. recompute the *actual* COM for each, by pinning the EQ search to that one
     operating point (LOCAL_SEARCH=0 over a 1x1x1x1 grid),
  3. report the FOM->COM rank correlation, the COM spread over the top-K, and the
     COM regret of trusting the FOM argmax.

Interpretation:
  low regret + high rank correlation -> pruning in FOM space is safe by construction
  high regret or low correlation     -> the adaptive search has a real tail risk,
                                        and this is where it comes from

Cost: one COM evaluation per probed candidate, plus one full-grid sweep to supply
the log. Budget it for a chosen handful of channels, not a whole corpus.

Usage:
    python fom_com_probe.py <config.xlsx> <thru.s4p> [--fext a.s4p ...] [--next n.s4p ...]
                            [--sweep-dir sweep_results] [--method full_grid]
                            [--top-k 10] [--out fom_com_probe.csv]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import com  # assembled COM module
from sweep_compare import _extract_case, apply_grid_reduction

PROBE_HEADER = ['fom_rank', 'gffe_index', 'ctle_index', 'lp_index', 'txffe_index',
                'tx_taps', 'FOM_dB', 'COM_dB', 'dFOM_vs_rank1', 'dCOM_vs_rank1',
                'wall_s']


def _parse_taps(s):
    """'[1 1 1 1 2]' -> [1, 1, 1, 1, 2]."""
    return [int(float(x)) for x in str(s).strip().strip('"').strip('[]').split()]


def _load_candidates(log_csv, top_k):
    """Top-K evaluated candidates by candidate_FOM, best first."""
    if not os.path.isfile(log_csv):
        raise SystemExit(f'sweep log not found: {log_csv}\n'
                         f'Run sweep_compare.py first to produce it.')
    with open(log_csv, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('evaluated') == '1']
    rows = [r for r in rows if _is_finite(r.get('candidate_FOM'))]
    if not rows:
        raise SystemExit(f'no evaluated candidates with a finite FOM in {log_csv}')
    rows.sort(key=lambda r: -float(r['candidate_FOM']))
    return rows[:top_k]


def _is_finite(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _pin_index(param, fields, i):
    """Pin parallel per-setting arrays to element i, keeping them index-aligned.

    Arrays shorter than i+1 are scalars shared across all settings — left alone.
    """
    for fld in fields:
        if not hasattr(param, fld):
            continue
        v = np.asarray(getattr(param, fld)).ravel()
        if v.size > i:
            setattr(param, fld, v[i:i + 1])


def _pinned_build(orig_build, cand):
    """Restrict OptFom_Build_TXFFE to the single TX-FFE row the candidate used.

    Pinning here rather than by collapsing param.tx_ffe_c*_values to one value
    each is deliberate: OptFom_Build_TXFFE treats a tap that is single-valued
    *and* zero as an absent leading tap and decrements the cursor position
    (com.py:3503-3517). Collapsing the value arrays would therefore silently
    move the cursor and change the TXFFE matrix shape. Slicing the built matrix
    leaves `cur`, precursor/postcursor indices and the cursor vector untouched.
    """
    def build(param):
        m, cur, sweep_idx, full_idx, cursor_vec = orig_build(param)
        tk = int(cand['txffe_index'])
        if not 0 <= tk < m.shape[0]:
            raise RuntimeError(f'txffe_index {tk} out of range for a '
                               f'{m.shape[0]}-row TXFFE grid — the probe grid does '
                               f'not match the one the log was written against')
        want = _parse_taps(cand['tx_taps'])
        got = [int(x) for x in np.asarray(full_idx[tk]).ravel()]
        if got != want:
            raise RuntimeError(f'TXFFE row {tk} is {got}, log says {want} — grid '
                               f'mismatch (check max_ctle / max_tap_vals)')
        return (m[tk:tk + 1], cur, sweep_idx,
                np.asarray(full_idx)[tk:tk + 1], np.asarray(cursor_vec)[tk:tk + 1])
    return build


def probe(config, thru, fext=(), next_=(), sweep_dir='sweep_results',
          method='full_grid', top_k=10, out_csv=None):
    files = [thru, *fext, *next_]
    out_csv = out_csv or os.path.join(sweep_dir, 'fom_com_probe.csv')

    # The EQ indices in the log are relative to the reduced grid the sweep ran on,
    # so reproduce that reduction before pinning.
    meta = {}
    summary_path = os.path.join(sweep_dir, 'summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            meta = json.load(f)
    max_ctle, max_tap_vals = meta.get('max_ctle'), meta.get('max_tap_vals')

    cands = _load_candidates(os.path.join(sweep_dir, f'{method}_log.csv'), top_k)
    print(f'Probing top {len(cands)} of the {method} log '
          f'(max_ctle={max_ctle}, max_tap_vals={max_tap_vals})', flush=True)

    orig_read = com.read_ParamConfigFile
    orig_build = com.OptFom_Build_TXFFE
    rows = []
    try:
        for rank, cand in enumerate(cands, 1):
            def patched_read(cf, OP, _c=cand):
                param, OP = orig_read(cf, OP)
                param.LOCAL_SEARCH = 0       # nothing to prune on a 1-point grid
                param.NonZeroLSMethod = 0
                apply_grid_reduction(param, max_ctle, max_tap_vals)
                _pin_index(param, ('ctle_gdc_values', 'CTLE_fp1', 'CTLE_fp2', 'CTLE_fz'),
                           int(_c['ctle_index']))
                _pin_index(param, ('g_DC_HP_values',), int(_c['lp_index']))
                _pin_index(param, ('cursor_gain',), int(_c['gffe_index']))
                return param, OP

            com.read_ParamConfigFile = patched_read
            com.OptFom_Build_TXFFE = _pinned_build(orig_build, cand)

            fom = float(cand['candidate_FOM'])
            print(f'\n=== rank {rank}/{len(cands)}: FOM={fom:.4f} dB  '
                  f'ctle={cand["ctle_index"]} lp={cand["lp_index"]} '
                  f'gffe={cand["gffe_index"]} taps={cand["tx_taps"]} ===', flush=True)

            t0 = time.time()
            results = com._run_com(config, len(fext), len(next_), files, export_mat=False)
            dt = time.time() - t0

            r = _extract_case(results)
            com_db = float(getattr(r, 'COM_dB', float('nan'))) if r is not None else float('nan')
            print(f'  -> COM = {com_db:.4f} dB  ({dt:.1f}s)', flush=True)

            rows.append({'fom_rank': rank, 'gffe_index': cand['gffe_index'],
                         'ctle_index': cand['ctle_index'], 'lp_index': cand['lp_index'],
                         'txffe_index': cand['txffe_index'], 'tx_taps': cand['tx_taps'],
                         'FOM_dB': fom, 'COM_dB': com_db, 'wall_s': round(dt, 1)})
    finally:
        com.read_ParamConfigFile = orig_read
        com.OptFom_Build_TXFFE = orig_build

    fom1, com1 = rows[0]['FOM_dB'], rows[0]['COM_dB']
    for r in rows:
        r['dFOM_vs_rank1'] = round(r['FOM_dB'] - fom1, 6)
        r['dCOM_vs_rank1'] = round(r['COM_dB'] - com1, 6)

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PROBE_HEADER)
        w.writeheader()
        w.writerows(rows)

    stats = _summarise(rows)
    stats.update({'config': os.path.basename(config), 'thru': os.path.basename(thru),
                  'method': method, 'top_k': len(rows), 'probe_csv': os.path.basename(out_csv)})
    with open(os.path.join(sweep_dir, 'fom_com_probe_summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    _report(stats, out_csv)
    return stats


def _summarise(rows):
    fom = np.array([r['FOM_dB'] for r in rows], dtype=float)
    com_db = np.array([r['COM_dB'] for r in rows], dtype=float)
    ok = np.isfinite(com_db)

    out = {'n_probed': int(len(rows)), 'n_com_failed': int((~ok).sum())}
    if ok.sum() < 2:
        out['note'] = 'too few successful COM evaluations to correlate'
        return out

    f, c = fom[ok], com_db[ok]
    from scipy.stats import spearmanr
    rho, p = spearmanr(f, c)

    best_i = int(np.argmax(c))
    out.update({
        'spearman_rho_FOM_vs_COM': round(float(rho), 4),
        'spearman_p': float(p),
        'COM_at_FOM_argmax_dB': round(float(c[0]), 6),
        'best_COM_in_topK_dB': round(float(c[best_i]), 6),
        # How much COM the search gives up by trusting FOM's ranking. This is the
        # number that matters: it bounds what *any* FOM-driven search can lose
        # inside the top-K, adaptive or not.
        'COM_regret_dB': round(float(c[best_i] - c[0]), 6),
        'COM_spread_topK_dB': round(float(c.max() - c.min()), 6),
        'FOM_spread_topK_dB': round(float(f.max() - f.min()), 6),
        'fom_argmax_is_com_argmax': bool(best_i == 0),
        'com_argmax_fom_rank': int(np.flatnonzero(ok)[best_i] + 1),
    })
    return out


def _report(s, out_csv):
    print('\n' + '=' * 68)
    print(f'FOM->COM probe: top {s["n_probed"]} candidates')
    if s.get('n_com_failed'):
        print(f'  !! {s["n_com_failed"]} COM evaluation(s) failed (NaN)')
    if 'spearman_rho_FOM_vs_COM' not in s:
        print(f'  {s.get("note", "")}')
        print('=' * 68)
        return
    print(f'  Spearman rho(FOM, COM)   = {s["spearman_rho_FOM_vs_COM"]:+.4f}  '
          f'(p={s["spearman_p"]:.3g})')
    print(f'  FOM spread over top-K    = {s["FOM_spread_topK_dB"]:.4f} dB')
    print(f'  COM spread over top-K    = {s["COM_spread_topK_dB"]:.4f} dB')
    print(f'  COM at FOM argmax        = {s["COM_at_FOM_argmax_dB"]:.4f} dB')
    print(f'  best COM in top-K        = {s["best_COM_in_topK_dB"]:.4f} dB '
          f'(FOM rank {s["com_argmax_fom_rank"]})')
    print(f'  COM regret               = {s["COM_regret_dB"]:.4f} dB')
    if s['fom_argmax_is_com_argmax']:
        print('  -> FOM argmax IS the COM argmax within the top-K.')
    else:
        print('  -> FOM argmax is NOT the COM argmax: FOM ranking is not '
              'COM-faithful here.')
    print(f'  wrote {out_csv}')
    print('=' * 68)


def main():
    ap = argparse.ArgumentParser(
        description='Recompute true COM for the top-K FOM candidates from a sweep log')
    ap.add_argument('config')
    ap.add_argument('thru')
    ap.add_argument('--fext', nargs='*', default=[])
    ap.add_argument('--next', nargs='*', default=[], dest='next_')
    ap.add_argument('--sweep-dir', default='sweep_results',
                    help='directory holding <method>_log.csv and summary.json')
    ap.add_argument('--method', default='full_grid',
                    choices=['full_grid', 'legacy', 'adaptive'],
                    help='which sweep log to draw candidates from (default: full_grid)')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--out', default=None, help='output CSV (default: <sweep-dir>/fom_com_probe.csv)')
    args = ap.parse_args()
    probe(args.config, args.thru, args.fext, args.next_, sweep_dir=args.sweep_dir,
          method=args.method, top_k=args.top_k, out_csv=args.out)


if __name__ == '__main__':
    main()
