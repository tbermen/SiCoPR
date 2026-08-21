"""Guard the results workbook against silently dropping columns.

Built from a real miss. The correlation report claimed the Tx FFE taps agreed
with MATLAB on all 208 cases, but `TXLE_taps_1..4` was never written into
results.xls at all -- so the claim was about a comparison that had not actually
been made. The same was true of the mixed-mode ERL variants and Pre2Pmax, which
a hand-maintained NOTES list described as "not computed" long after they were.

The failure mode is not a wrong number. It is a column that is absent or blank
while a document asserts something about it. These tests fail when that happens:

  * `_txffe_canonical` maps a trimmed vector onto [c(-2) c(-1) c(0) c(1)]
    correctly, including the fully-trimmed and swept-precursor cases;
  * every column the review compares is populated on a real exported row.

The second test needs a completed run; it skips when there is none, rather than
passing vacuously.
"""
import glob
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from export_results import _txffe_canonical, build_row  # noqa: E402
from matlab_compare import OUTDIR  # noqa: E402

# Columns the correlation review makes a claim about. If a column is listed here
# it must be populated on a real row -- adding a claim to the document without
# adding the column here is the mistake this test exists to catch.
CLAIMED = [
    'COM_dB', 'FOM', 'itick', 'VEO_mV', 'VEC_dB',
    'ERL11', 'ERL11_CD', 'ERL11_DC', 'ERL11_CC',
    'ERL22', 'ERL22_CD', 'ERL22_DC', 'ERL22_CC',
    'TXLE_taps_1', 'TXLE_taps_2', 'TXLE_taps_3', 'TXLE_taps_4',
    'Pre2Pmax', 'CTLE_DC_gain_dB',
]


@pytest.mark.parametrize('taps,expect', [
    # fully trimmed: unity Tx FFE, only the cursor survives
    ([1.0], (0.0, 0.0, 1.0, 0.0)),
    # cursor + one postcursor
    ([0.9, -0.1], (0.0, 0.0, 0.9, -0.1)),
    # one precursor kept
    ([-0.05, 0.9, -0.05], (0.0, -0.05, 0.9, -0.05)),
    # both precursors kept -- the swept-c(-2) shape the wXtalk reference uses
    ([0.02, -0.05, 0.88, -0.05], (0.02, -0.05, 0.88, -0.05)),
    # a bare scalar, not a vector
    (1.0, (0.0, 0.0, 1.0, 0.0)),
    (None, None),
])
def test_txffe_canonical(taps, expect):
    got = _txffe_canonical(taps)
    if expect is None:
        assert got is None
        return
    assert (got['c(-2)'], got['c(-1)'], got['c(0)'], got['c(1)']) == expect


def test_cursor_is_n_post_from_the_end():
    """The cursor position is defined by n_post, not by the vector length.

    OptFom_Build_TXFFE trims leading precursors only, so length varies while the
    distance from the end does not. Indexing from the front would put the cursor
    in the wrong slot exactly when precursors are trimmed -- which is the common
    case, and would have gone unnoticed.
    """
    for n_post in (1, 2, 3):
        t = [0.1] * 2 + [0.8] + [0.05] * n_post
        assert _txffe_canonical(t, n_post=n_post)['c(0)'] == 0.8


def _any_completed_row():
    man_p = os.path.join(OUTDIR, 'manifest.json')
    if not os.path.isfile(man_p):
        return None
    man = json.load(open(man_p, encoding='utf-8'))
    # prefer a directory produced with --modal-erl, which is what populates the
    # mixed-mode ERL columns the review compares
    for sub in ('cases_txffesweep_modalerl_mr2', 'cases_modalerl'):
        d = os.path.join(OUTDIR, sub)
        for p in sorted(glob.glob(os.path.join(d, 'wXtalk_*.result.json'))):
            rr = json.load(open(p, encoding='utf-8'))
            if not (rr.get('ok') and rr.get('cases')):
                continue
            cid = os.path.basename(p).split('.')[0]
            case = next((c for c in man['cases'] if c['case_id'] == cid), None)
            if case:
                return man['headers'][case['cond']], case, rr
    return None


def test_claimed_columns_are_populated():
    found = _any_completed_row()
    if found is None:
        pytest.skip('no completed --modal-erl run to export from')
    headers, case, rr = found

    from export_results import _sheet_params
    from matlab_compare import CONFIGS, TAB_CONFIG
    cfg = _sheet_params(os.path.join(CONFIGS, TAB_CONFIG[case['test']]))
    row = build_row(headers, case, rr['cases'][0], cfg,
                    rr.get('wall_s', 0) / 60.0, None)

    idx, seen = {}, set()
    for i, h in enumerate(headers):
        if h is not None and h not in seen:
            seen.add(h)
            idx[h] = i

    missing = [c for c in CLAIMED if c not in idx]
    blank = [c for c in CLAIMED if c in idx and row[idx[c]] in (None, '')]
    assert not missing, 'columns absent from the reference header: %s' % missing
    assert not blank, (
        'columns the correlation review compares, but which this export leaves '
        'blank: %s -- either populate them or stop claiming they agree' % blank)
