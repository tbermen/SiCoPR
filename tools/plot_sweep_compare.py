"""plot_sweep_compare.py — Static (matplotlib) figures comparing the EQ-search
methods (full grid / legacy local search / adaptive local search) from the logs
written by sweep_compare.py.

Figures written to <dir>:
    convergence.png      best-so-far FOM vs candidate count (quality + speed)
    effort.png           evaluated vs skipped candidates per method
    coverage.png         EQ-settings-vs-FOM: 2-D TX-FFE tap scatter, one panel
                         per method over the full-grid FOM surface
    parallel_coords.png  all tap dims + CTLE + LP colored by FOM
    adaptive_radius.png  adaptive-radius / L1 / L2 trajectory (adaptive only)
    summary_table.png    + summary.csv  per-method metrics

Usage:  python plot_sweep_compare.py [sweep_results_dir]
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['full_grid', 'legacy', 'adaptive']
COLORS = {'full_grid': '#999999', 'legacy': '#1f77b4', 'adaptive': '#d62728'}


def _parse_taps(s):
    return [float(v) for v in re.findall(r'-?\d+\.?\d*', s)]


def load_log(path):
    """Load a sweep log CSV into a dict of numpy arrays (+ tap matrix)."""
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None
    taps = np.array([_parse_taps(r['tx_taps']) for r in rows], dtype=float)
    return {
        'method': rows[0]['method'],
        'ctle_index': np.array([float(r['ctle_index']) for r in rows]),
        'lp_index': np.array([float(r['lp_index']) for r in rows]),
        'cand_fom': np.array([float(r['candidate_FOM']) for r in rows]),
        'best_fom': np.array([float(r['best_FOM']) for r in rows]),
        'evaluated': np.array([int(r['evaluated']) for r in rows]),
        'eval_count': np.array([int(r['eval_count']) for r in rows]),
        'taps': taps,
    }


def _eval_mask(d):
    m = (d['evaluated'] == 1) & np.isfinite(d['cand_fom'])
    return m


def plot_convergence(logs, out):
    plt.figure(figsize=(7, 4.5))
    for name in METHODS:
        d = logs.get(name)
        if d is None:
            continue
        m = d['evaluated'] == 1
        x = d['eval_count'][m]
        y = d['best_fom'][m]
        order = np.argsort(x)
        plt.plot(x[order], y[order], label=name, color=COLORS[name], lw=1.8)
    # full-grid optimum as reference line
    fg = logs.get('full_grid')
    if fg is not None:
        opt = np.nanmax(fg['cand_fom'][_eval_mask(fg)])
        plt.axhline(opt, ls='--', color='k', lw=0.8, label=f'full-grid optimum ({opt:.2f} dB)')
    plt.xlabel('candidates evaluated'); plt.ylabel('best-so-far FOM (dB)')
    plt.title('Convergence: FOM vs search effort')
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out, 'convergence.png'), dpi=130); plt.close()


def plot_effort(logs, summary, out):
    names = [n for n in METHODS if logs.get(n) is not None]
    ev = [int((logs[n]['evaluated'] == 1).sum()) for n in names]
    sk = [int((logs[n]['evaluated'] == 0).sum()) for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4.5))
    plt.bar(x, ev, 0.6, label='evaluated', color='#2ca02c')
    plt.bar(x, sk, 0.6, bottom=ev, label='skipped (search-pruned)', color='#cccccc')
    full_eval = ev[0] if names and names[0] == 'full_grid' else max(ev) if ev else 1
    totals = [e + s for e, s in zip(ev, sk)]
    plt.ylim(top=max(totals + [1]) * 1.18)
    for i, (e, s) in enumerate(zip(ev, sk)):
        pct = 100.0 * e / max(full_eval, 1)
        plt.text(i, e + s, f'{e} eval\n{pct:.0f}% of grid', ha='center', va='bottom', fontsize=8)
    plt.xticks(x, names); plt.ylabel('TX-FFE candidates considered')
    plt.title('Search effort: evaluated vs skipped', pad=24)
    plt.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'effort.png'), dpi=130); plt.close()


def _top2_tap_dims(fg):
    var = fg['taps'].var(axis=0)
    if len(var) < 2:
        return 0, min(1, fg['taps'].shape[1] - 1)
    idx = np.argsort(var)[::-1]
    return int(idx[0]), int(idx[1])


def plot_coverage(logs, out):
    fg = logs.get('full_grid')
    if fg is None:
        return
    i, j = _top2_tap_dims(fg)
    names = [n for n in METHODS if logs.get(n) is not None]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6), squeeze=False)
    vmin = np.nanmin(fg['cand_fom'][_eval_mask(fg)])
    vmax = np.nanmax(fg['cand_fom'][_eval_mask(fg)])
    for ax, name in zip(axes[0], names):
        d = logs[name]
        # faint full-grid FOM surface as background
        bg = _eval_mask(fg)
        ax.scatter(fg['taps'][bg, i], fg['taps'][bg, j], c=fg['cand_fom'][bg],
                   cmap='viridis', s=10, alpha=0.18, vmin=vmin, vmax=vmax)
        # skipped (search-pruned) first, then evaluated on top
        sk = d['evaluated'] == 0
        ax.scatter(d['taps'][sk, i], d['taps'][sk, j], facecolors='none',
                   edgecolors='red', s=12, lw=0.4, alpha=0.35, label='skipped')
        ev = _eval_mask(d)
        sc = ax.scatter(d['taps'][ev, i], d['taps'][ev, j], c=d['cand_fom'][ev],
                        cmap='viridis', s=28, edgecolor='k', lw=0.4, vmin=vmin, vmax=vmax,
                        zorder=4)
        # optimum star
        if ev.any():
            k = np.nanargmax(np.where(ev, d['cand_fom'], np.nan))
            ax.scatter([d['taps'][k, i]], [d['taps'][k, j]], marker='*', s=240,
                       color='gold', edgecolor='k', zorder=5)
        ax.set_title(f'{name}  ({int(ev.sum())} eval, {int(sk.sum())} skip)')
        ax.set_xlabel(f'TX-FFE tap[{i}] index'); ax.set_ylabel(f'TX-FFE tap[{j}] index')
    fig.colorbar(sc, ax=axes[0].tolist(), label='candidate FOM (dB)', shrink=0.8)
    fig.suptitle('EQ-settings vs FOM: TX-FFE tap-space coverage (background = full-grid surface)')
    fig.savefig(os.path.join(out, 'coverage.png'), dpi=130, bbox_inches='tight'); plt.close(fig)


def plot_parallel_coords(logs, out):
    names = [n for n in METHODS if logs.get(n) is not None]
    fg = logs.get('full_grid')
    ntap = (fg or logs[names[0]])['taps'].shape[1]
    dims = [f't{k}' for k in range(ntap)] + ['ctle', 'lp']
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6), squeeze=False)
    # shared color scale from full grid
    base = fg if fg is not None else logs[names[0]]
    vmin = np.nanmin(base['cand_fom'][_eval_mask(base)])
    vmax = np.nanmax(base['cand_fom'][_eval_mask(base)])
    for ax, name in zip(axes[0], names):
        d = logs[name]
        ev = _eval_mask(d)
        X = np.column_stack([d['taps'][ev], d['ctle_index'][ev], d['lp_index'][ev]]).astype(float)
        if X.shape[0] == 0:
            continue
        # normalise each dim to [0,1] for display
        lo = X.min(axis=0); hi = X.max(axis=0); rng = np.where(hi > lo, hi - lo, 1.0)
        Xn = (X - lo) / rng
        fom = d['cand_fom'][ev]
        order = np.argsort(fom)  # draw best last (on top)
        cmap = plt.get_cmap('viridis')
        for r in order:
            ax.plot(range(len(dims)), Xn[r], color=cmap((fom[r] - vmin) / max(vmax - vmin, 1e-9)),
                    alpha=0.25, lw=0.6)
        ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, fontsize=7)
        ax.set_yticks([]); ax.set_title(f'{name}')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=axes[0].tolist(), label='candidate FOM (dB)', shrink=0.8)
    fig.suptitle('EQ settings (parallel coordinates) colored by FOM')
    fig.savefig(os.path.join(out, 'parallel_coords.png'), dpi=130, bbox_inches='tight'); plt.close(fig)


def plot_adaptive_radius(out):
    path = os.path.join(out, 'adaptive_radius_log.csv')
    if not os.path.isfile(path):
        return
    rows = list(csv.DictReader(open(path, newline='')))
    if not rows:
        return
    def col(name):
        return np.array([float(r[name]) if r.get(name) not in (None, '', 'nan') else np.nan
                         for r in rows])
    it = col('iter'); ar = col('adaptive_radius'); dr = col('deterministic_radius')
    l1 = col('L1_w'); l2 = col('L2_w'); skip = col('skip_it')
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(it, ar, label='adaptive_radius', color='#d62728')
    a1.plot(it, dr, label='deterministic_radius', color='#1f77b4', ls='--')
    a1.set_ylabel('radius'); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a1.set_title('Adaptive local search: radius adaptation')
    kept = skip == 0
    a2.scatter(it[kept], l1[kept], s=10, color='#2ca02c', label='L1_w (evaluated)')
    a2.scatter(it[~kept], l1[~kept], s=10, color='red', label='L1_w (skipped)')
    a2.plot(it, ar, color='#d62728', lw=0.8, alpha=0.7, label='adaptive_radius')
    a2.set_xlabel('iteration'); a2.set_ylabel('weighted L1 distance')
    a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'adaptive_radius.png'), dpi=130); plt.close(fig)


def write_summary_table(summary, out):
    methods = summary.get('methods', {})
    cols = ['COM_dB', 'best_FOM_dB', 'n_evaluated', 'n_skipped',
            'pct_of_full_grid_evaluated', 'speedup_vs_full_grid', 'wall_s']
    names = [n for n in METHODS if n in methods]
    # CSV
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method'] + cols)
        for n in names:
            w.writerow([n] + [methods[n].get(c, '') for c in cols])
    # PNG table
    fig, ax = plt.subplots(figsize=(1.7 * (len(cols) + 1), 0.6 * (len(names) + 1) + 0.6))
    ax.axis('off')
    cell = [[f'{methods[n].get(c, ""):g}' if isinstance(methods[n].get(c), (int, float))
             else str(methods[n].get(c, '')) for c in cols] for n in names]
    tbl = ax.table(cellText=cell, rowLabels=names, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title('EQ-search method comparison', pad=14)
    fig.savefig(os.path.join(out, 'summary_table.png'), dpi=130, bbox_inches='tight'); plt.close(fig)


def main(out='sweep_results'):
    logs = {}
    for n in METHODS:
        p = os.path.join(out, f'{n}_log.csv')
        if os.path.isfile(p):
            logs[n] = load_log(p)
    if not logs:
        print(f'No *_log.csv found in {out}/ — run sweep_compare.py first.')
        return
    summary = {}
    sp = os.path.join(out, 'summary.json')
    if os.path.isfile(sp):
        summary = json.load(open(sp))

    plot_convergence(logs, out)
    plot_effort(logs, summary, out)
    plot_coverage(logs, out)
    plot_parallel_coords(logs, out)
    plot_adaptive_radius(out)
    if summary:
        write_summary_table(summary, out)
    print(f'Wrote figures to {out}/: convergence, effort, coverage, parallel_coords, '
          f'adaptive_radius, summary_table')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'sweep_results')
