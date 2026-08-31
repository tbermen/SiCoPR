"""Guard against SILENT SEARCH-SPACE COLLAPSE.

The Tx FFE defect (docs/TXFFE_SWEEP_ROOT_CAUSE.md) cost weeks and produced a
wrong conclusion about the MATLAB reference. Its shape:

    A            B      C                  D
    c(-1)        0      [ -0.34:.02:0]     [min:step:max]

Both engines read the cell immediately RIGHT of the label — column B, the `0`.
The sweep range sat one further over as a template. Nothing errored. The Tx FFE
search dimension silently collapsed from 198 candidates to 1, the run looked
healthy, and 10 of 208 cases quietly disagreed with MATLAB for months.

Nothing in the suite could have caught that, because every existing test asks
"does this function compute the right answer?" and none asks "is the optimiser
actually being given anything to search?".

Three checks here:

1. SEARCH DIMENSIONS ARE PINNED. The five dimensions optimize_fom loops over are
   recorded per shipped config. If a parser change, a config edit or a MATLAB
   version switch collapses one, the number moves and the test fails. This is the
   check that would have caught the Tx FFE defect on day one.

2. ADJACENT-TEMPLATE LEDGER. Any config keyword whose value cell is a scalar
   while a real MATLAB range literal sits in the next cell along is reported.
   Known instances are listed in ADJACENT_KNOWN with a note; a NEW one fails.

3. KEYWORD PARITY. Every keyword the MATLAB reference reads must also be read by
   sicopr.py. A keyword MATLAB reads and the port ignores is a setting the user can
   set and silently have discarded — the same failure class one level up.

Run: python tests/test_config_search_space.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import glob
import io
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

import sicopr  # noqa: E402

CONFIG_DIR = os.path.join(_ROOT, 'tests', '1_IEEE_802p3dj_COM_Spreadsheets')
MATLAB_REF = os.path.join(_ROOT, 'matlab', 'com_ieee8023_4p15p0.m')

# ---------------------------------------------------------------------------
# 1. Expected search-space size, per config, for the five dimensions
#    optimize_fom actually loops over.
#
#    RESOLVED 2026-08-24. The MATLAB author confirmed that the with-crosstalk
#    reference run DID sweep the Tx FFE; the workbook that captured those sweeps
#    was simply not the one originally shared. The *_sweep_TxFFE configs carry
#    the ranges in the Setting column and define a 1584-point grid.
#
#    Both shapes are therefore legitimate and both are pinned, keyed by filename:
#    a plain config must stay single-point and a sweep config must stay 1584. If
#    either moves, the correlation has to be re-stated rather than the headline
#    numbers shifting unnoticed -- which is the whole point of this file.
# ---------------------------------------------------------------------------
EXPECTED_DIMS = {
    'cursor_gain (Gffe)': 1,     # 'crusor_gain' (MATLAB's typo) defaults to 0
    'ctle_gdc_values': 21,
    'g_DC_HP_values': 7,
    'Tx FFE grid': 1,            # overridden for *_sweep_TxFFE, see below
    'itick range': 49,
}

# Per-config overrides. The with-crosstalk reference used these.
EXPECTED_DIMS_SWEEP = dict(EXPECTED_DIMS, **{
    'Tx FFE grid': 1584,         # c(-1) 18 x c(-2) 8 x c(1) 11
})


def expected_for(base):
    """Which pinned search space this config is required to have."""
    return EXPECTED_DIMS_SWEEP if base.endswith('_sweep_TxFFE') else EXPECTED_DIMS

# Keyword -> why a scalar value with a range template beside it is expected here.
ADJACENT_KNOWN = {
    'c(-1)': 'Tx FFE sweep disabled in the supplied configs; the MATLAB '
             'reference run had it enabled (docs/TXFFE_SWEEP_ROOT_CAUSE.md)',
    'c(-2)': 'as c(-1)',
    'c(1)': 'as c(-1)',
}

_RANGE = re.compile(r'^\s*\[?\s*-?[\d.]+\s*:\s*-?[\d.]+\s*(:\s*-?[\d.]+\s*)?\]?\s*$')
_VECTOR = re.compile(r'^\s*\[[^\]]*[\s,][^\]]*\]\s*$')


def looks_like_sweep(v):
    """A real MATLAB range/vector literal, not the '[min:step:max]' placeholder."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s or s.lower().strip('[]') in ('min:step:max',):
        return False
    return bool(_RANGE.match(s) or _VECTOR.match(s))


def _bootstrap_OP():
    src = io.open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read().split('\n')
    a = next(i for i, l in enumerate(src) if l.strip() == 'OP = SimpleNamespace()')
    b = next(i for i, l in enumerate(src) if 'param, OP = read_ParamConfigFile' in l)
    ns = {'SimpleNamespace': SimpleNamespace}
    exec('\n'.join(l[4:] for l in src[a:b]), ns)
    return ns['OP']


def search_dims(path):
    param, OP = sicopr.read_ParamConfigFile(path, _bootstrap_OP())

    def n(v):
        return int(np.atleast_1d(np.asarray(v)).size)

    tsar = np.atleast_1d(np.asarray(param.ts_sample_adj_range)).ravel()
    txffe = sicopr.OptFom_Build_TXFFE(param)[0]
    return {
        'cursor_gain (Gffe)': n(getattr(param, 'cursor_gain', 0)),
        'ctle_gdc_values': n(param.ctle_gdc_values),
        'g_DC_HP_values': n(getattr(param, 'g_DC_HP_values', 0)),
        'Tx FFE grid': int(np.atleast_2d(txffe).shape[0]),
        'itick range': int(tsar[1] - tsar[0] + 1) if tsar.size >= 2 else 1,
    }


def adjacent_templates(path):
    out = []
    wb = openpyxl.load_workbook(path, data_only=True)
    for name in wb.sheetnames:
        ws = wb[name]
        for row in ws.iter_rows():
            for c in row:
                if not isinstance(c.value, str) or not c.value.strip():
                    continue
                val = ws.cell(c.row, c.column + 1).value
                nxt = ws.cell(c.row, c.column + 2).value
                if isinstance(val, (int, float)) and looks_like_sweep(nxt):
                    out.append((c.value.strip(), val, str(nxt).strip()))
    return out


def _strip_matlab_comments(text):
    """Drop % comments so commented-out xls_parameter calls are not counted."""
    out = []
    for line in text.split('\n'):
        i = line.find('%')
        while i > 0 and line[i - 1] == '\\':          # escaped percent
            i = line.find('%', i + 1)
        out.append(line if i < 0 else line[:i])
    return '\n'.join(out)



def _version_gated_keyword_check():
    """No config keyword may be read only inside a MATLAB-version branch.

    Ledger #12. A keyword read under `if _v416:` and nowhere else is honoured on
    one emulation path and quietly ignored on the other -- which is how a config
    that set `Overwrite Minimum Radius` under 4p15p0 had it discarded.
    """
    import ast
    fn_root = os.path.join(_ROOT, 'com_functions', 'fn')
    offenders = []
    gated_fns = 0
    for d in sorted(os.listdir(fn_root)):
        p = os.path.join(fn_root, d, 'py_impl.py')
        if not os.path.isfile(p):
            continue
        src = io.open(p, encoding='utf-8').read()
        if '_v416' not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        gated_fns += 1
        gates = [n for n in ast.walk(tree)
                 if isinstance(n, ast.If) and '_v416' in ast.dump(n.test)]
        gated_ids, inside = set(), set()
        for g in gates:
            for sub in ast.walk(g):
                gated_ids.add(id(sub))
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)                         and sub.value.id in ('param', 'OP')                         and isinstance(sub.ctx, ast.Load):
                    inside.add('%s.%s' % (sub.value.id, sub.attr))
        outside = set()
        for sub in ast.walk(tree):
            if id(sub) in gated_ids:
                continue
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)                     and sub.value.id in ('param', 'OP')                     and isinstance(sub.ctx, ast.Load):
                outside.add('%s.%s' % (sub.value.id, sub.attr))
        for nm in sorted(inside - outside):
            offenders.append('%s reads %s only inside a version gate' % (d, nm))

    check("version_gated_functions_were_scanned", gated_fns > 0,
          "no version-gated function found; the check is looking in the wrong place")
    check("no_config_keyword_is_read_on_only_one_version_path",
          not offenders,
          "a config keyword read under `if _v416:` and nowhere else is honoured "
          "on one emulation path and silently discarded on the other. That is "
          "ledger #12 exactly, and the keyword-parity check cannot see it "
          "because the keyword IS read. Read it on both paths: "
          + "; ".join(offenders))


def main():
    configs = sorted(glob.glob(os.path.join(CONFIG_DIR, '*.xlsx')))
    # ---- ledger #12: a keyword read on only ONE version path ----------------
    # Overwrite_Min_Radius was read inside `if _v416:` and nowhere else, so a
    # config that set it while emulating 4p15p0 had it SILENTLY DISCARDED. The
    # keyword-parity check below cannot see this: the keyword IS read, just not
    # on every path, so parity is satisfied.
    #
    # Static, so it runs before the config-workbook skip below and works on a
    # clone with no correlation data.
    _version_gated_keyword_check()

    if not configs:
        # The COM configuration workbooks are correlation INPUTS and are not
        # redistributed with the port (see README "What ships"). A clone without
        # them is the normal case for anyone who has not downloaded the IEEE
        # contributions, and it must not look like a broken repository: skip
        # cleanly rather than failing the suite.
        print('SKIP config_search_space: no config workbooks in %s' % CONFIG_DIR)
        print('     These are correlation inputs, not shipped with the port.')
        print('     See README.md for which IEEE 802.3dj contributions to download.')
        return 0

    # ---- 1. search dimensions -------------------------------------------
    for path in configs:
        base = os.path.basename(path).replace('.xlsx', '')
        try:
            dims = search_dims(path)
        except Exception as exc:
            check('search_dims_%s' % base, False, 'could not evaluate: %r' % (exc,))
            continue
        for key, expected in expected_for(base).items():
            got = dims.get(key)
            check('dim_%s_%s' % (key.split()[0].strip('()'), base),
                  got == expected,
                  'search dimension %r is %r, expected %r. A change here moves the '
                  'EQ search space and therefore every reported COM/FOM/itick; '
                  'update EXPECTED_DIMS deliberately and re-state the correlation.'
                  % (key, got, expected))

    # ---- 2. adjacent-template ledger -------------------------------------
    for path in configs:
        base = os.path.basename(path).replace('.xlsx', '')
        new = [t for t in adjacent_templates(path) if t[0] not in ADJACENT_KNOWN]
        check('no_new_adjacent_template_%s' % base, not new,
              'config keyword(s) %s hold a scalar while a sweep range sits in the '
              'next cell along. That is the Tx FFE failure mode: the search '
              'dimension collapses to one point with no error. Either enable the '
              'sweep or add it to ADJACENT_KNOWN with a reason.'
              % ', '.join('%s (value=%s, next=%s)' % t for t in new))

    # ---- 3. keyword parity ------------------------------------------------
    py = io.open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read()
    ml = _strip_matlab_comments(
        io.open(MATLAB_REF, encoding='utf-8', errors='replace').read())
    # The port reads keywords through several call forms -- __xls_param, the
    # package-block xp() wrapper, _load_zp() helpers, and plain list literals --
    # so enumerating call sites gives false positives. Ask the weaker but robust
    # question instead: does the keyword literal appear in sicopr.py at all?
    ml_kw = {k for k in
             re.findall(r"xls_parameter(?:_txffe)?\(\s*\w+\s*,\s*'([^']+)'", ml)}
    py_lower = py.lower()
    missing = sorted(k for k in ml_kw
                     if ("'%s'" % k).lower() not in py_lower
                     and ('"%s"' % k).lower() not in py_lower)
    check('keyword_parity_matlab_subset_of_python', not missing,
          'sicopr.py never reads %d keyword(s) that %s reads: %s. A keyword the '
          'reference honours and the port ignores is a setting the user can set '
          'and silently have discarded.'
          % (len(missing), os.path.basename(MATLAB_REF), ', '.join(missing)))

    return finish()


if __name__ == '__main__':
    sys.exit(main())
