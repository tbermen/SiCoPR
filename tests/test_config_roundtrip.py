"""GUI phase 0 — a config written by `gui.config_io` parses exactly like its source.

Nothing built on top of the config writer can be trusted unless this holds, so it
is the first thing written and the gate everything else waits behind.

For every configuration workbook in the repo:

  1. read it with `gui.config_io.read_config`
  2. write it back out unchanged
  3. hand BOTH files to the real engine (`read_ParamConfigFile`) and require the
     resulting `param` and `OP` to be field-for-field identical

Step 3 is the point. Comparing the two workbooks to each other would only prove
the writer is self-consistent; comparing what the ENGINE makes of them proves the
thing that actually matters.

Then the harder cases, each of which is a real hazard rather than a hypothetical:

  * a formula-backed keyword (`f_v`, `f_f`, `f_n` are computed from `f_b`)
    survives an untouched round trip with its CACHED value intact. openpyxl's
    save path cannot do this -- it keeps the formula and drops the number the
    engine reads -- which is why the writer edits the sheet XML directly.
  * changing a formula's precedent is REPORTED as stale rather than silently
    producing a workbook whose numbers contradict its own arithmetic.
  * a Tx FFE tap written as a sweep range string stays a string, and one written
    as a scalar stays a scalar. This distinction cost the project a long
    investigation once already (ledger: TxFFE config mismatch) -- the value
    column and the range template in the units column are NOT interchangeable,
    and a GUI that blurs them would reintroduce it.

    python tests/test_config_roundtrip.py
"""
import glob
import io
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, xcheck, finish  # noqa: E402
from gui.config_io import read_config, write_config  # noqa: E402

import sicopr  # noqa: E402

CONFIG_DIRS = [
    os.path.join(_ROOT, 'tests', '1_IEEE_802p3dj_COM_Spreadsheets'),
    os.path.join(_ROOT, 'tests', 'configs'),
    os.path.join(_ROOT, 'config'),
]


def _configs():
    out = []
    for d in CONFIG_DIRS:
        if os.path.isdir(d):
            out += sorted(glob.glob(os.path.join(d, '*.xlsx')))
    return [p for p in out if not os.path.basename(p).startswith('~$')]


def _bootstrap_OP():
    """The OP the driver hands to read_ParamConfigFile, lifted from the driver
    itself so this test cannot drift from it. Same trick as
    tests/test_config_search_space.py."""
    src = io.open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read().splitlines()
    a = next(i for i, l in enumerate(src) if l.strip() == 'OP = SimpleNamespace()')
    b = next(i for i, l in enumerate(src) if 'param, OP = read_ParamConfigFile' in l)
    ns = {'SimpleNamespace': SimpleNamespace}
    exec(chr(10).join(l[4:] for l in src[a:b]), ns)
    return ns['OP']


def _parse(path):
    """What the engine makes of a workbook: (param, OP) as plain dicts."""
    param, OP = sicopr.read_ParamConfigFile(path, _bootstrap_OP())
    return vars(param), vars(OP)


def _equal(x, y, path, bad):
    """Deep exact equality. param carries nested namespaces (param.PKG), lists
    and arrays, so a flat `x != y` both misses differences and raises on the
    ambiguous-truth-value of an array."""
    if isinstance(x, SimpleNamespace) or isinstance(y, SimpleNamespace):
        if not (isinstance(x, SimpleNamespace) and isinstance(y, SimpleNamespace)):
            bad.append('%s: %s vs %s' % (path, type(x).__name__, type(y).__name__))
            return
        _diff(vars(x), vars(y), path, bad)
        return
    if isinstance(x, dict) and isinstance(y, dict):
        _diff(x, y, path, bad)
        return
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            bad.append('%s: length %d vs %d' % (path, len(x), len(y)))
            return
        for i, (u, v) in enumerate(zip(x, y)):
            _equal(u, v, '%s[%d]' % (path, i), bad)
        return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            bad.append('%s: shape %s vs %s' % (path, x.shape, y.shape))
        elif x.dtype.kind in 'fc' and y.dtype.kind in 'fc':
            if not np.array_equal(x, y, equal_nan=True):
                bad.append('%s: values differ' % path)
        elif not np.array_equal(x, y):
            bad.append('%s: values differ' % path)
        return
    if isinstance(x, float) and isinstance(y, float):
        if not (x == y or (np.isnan(x) and np.isnan(y))):
            bad.append('%s: %r vs %r' % (path, x, y))
        return
    if x is not y and x != y:
        bad.append('%s: %r vs %r' % (path, x, y))


def _diff(a, b, label, bad=None):
    """Field-by-field difference between two engine-parsed structs."""
    top = bad is None
    if top:
        bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append('%s.%s present in only one' % (label, k))
            continue
        try:
            _equal(a[k], b[k], '%s.%s' % (label, k), bad)
        except Exception as e:                       # noqa: BLE001
            bad.append('%s.%s comparison failed: %s' % (label, k, e))
    return bad


configs = _configs()
if not configs:
    # Configuration workbooks are IEEE contributions and are not redistributed
    # with the repository, so a fresh clone legitimately has none. The contract
    # CI enforces is that such a test SKIPS and says what is missing -- failing
    # here would make every public clone red for want of data it cannot have.
    print("SKIP: no configuration workbooks found under %s"
          % [os.path.relpath(d, _ROOT) for d in CONFIG_DIRS])
    print("      This test needs a COM config .xlsx; see README section 1.")
    sys.exit(0)

tmp = tempfile.mkdtemp(prefix='cfg_roundtrip_')
try:
    # ---------------------------------------------------------------- 1. identity
    failures = []
    for src in configs:
        dst = os.path.join(tmp, 'rt_' + os.path.basename(src))
        try:
            warns = write_config(src, dst, {})
            if warns:
                failures.append('%s: an unchanged rewrite produced warnings: %s'
                                % (os.path.basename(src), warns))
            pa, oa = _parse(src)
            pb, ob = _parse(dst)
            failures += ['%s: %s' % (os.path.basename(src), m)
                         for m in _diff(pa, pb, 'param') + _diff(oa, ob, 'OP')]
        except Exception as e:                       # noqa: BLE001
            failures.append('%s: %s: %s' % (os.path.basename(src),
                                            type(e).__name__, e))

    check("an_unchanged_rewrite_parses_identically",
          not failures,
          "the engine sees a different configuration after a no-op round trip, so "
          "the writer is damaging the workbook:\n     " + "\n     ".join(failures[:25]))

    # ------------------------------------------------- 2. formula cells survive
    probe = configs[0]
    doc = read_config(probe)
    formula_kws = [k for k, v in doc.keywords.items() if v.is_formula]
    check("the_corpus_contains_a_formula_backed_keyword", formula_kws,
          "no keyword in %s is backed by a formula, so the hazard this writer "
          "exists to avoid is not represented and the next two checks prove "
          "nothing" % os.path.basename(probe))

    if formula_kws:
        dst = os.path.join(tmp, 'formula_' + os.path.basename(probe))
        write_config(probe, dst, {})
        after = read_config(dst)
        lost = [k for k in formula_kws
                if after.keywords.get(k) is None
                or after.keywords[k].value is None
                or after.keywords[k].value != doc.keywords[k].value]
        check("formula_backed_values_survive_a_round_trip",
              not lost,
              "these keywords are computed by a spreadsheet formula and lost their "
              "cached value on the way through. The engine loads with "
              "data_only=True, so it would now read None and the config would be "
              "silently broken: %s" % lost)

        kept = [k for k in formula_kws if not after.keywords[k].is_formula]
        check("formulas_themselves_survive_a_round_trip",
              not kept,
              "these cells were formulas in the source and are literals in the "
              "copy. The engine does not care, but the workbook has stopped "
              "tracking its own inputs: %s" % kept)

    # ------------------------------------------- 3. stale dependency is reported
    if formula_kws:
        prec = None
        for k, v in doc.keywords.items():
            if v.is_formula and 'B3' in (v.formula or ''):
                prec = k
                break
        b3_owner = next((k for k, v in doc.keywords.items() if v.coord == 'B3'), None)
        if prec and b3_owner:
            dst = os.path.join(tmp, 'stale_' + os.path.basename(probe))
            warns = write_config(probe, dst, {b3_owner: 999.0})
            check("editing_a_formula_precedent_is_reported_as_stale",
                  any(w.startswith('STALE:') for w in warns),
                  "changing %s (%s) leaves the formula-backed keyword %s holding a "
                  "cached number computed from the OLD value, and the engine reads "
                  "cached numbers. That has to be reported, not swallowed. "
                  "warnings=%s" % (b3_owner, 'B3', prec, warns))
        else:
            xcheck("editing_a_formula_precedent_is_reported_as_stale", False,
                   "this corpus has no formula referencing B3 with a named owner; "
                   "the stale-detection path is exercised by unit values instead")

    # ------------------------------------------ 4. Tx FFE value vs range string
    #
    # A tap set to a scalar means "use this value". The same tap set to a MATLAB
    # range string means "sweep it". The units column holds a range template even
    # when the value is a scalar, so the two look alike at a glance and do not
    # behave alike at all.
    tap_names = [k for k in doc.keywords if k.strip().lower().startswith('c(')]
    check("tx_ffe_tap_keywords_are_addressable", tap_names,
          "no c(-1)/c(1)-style Tx FFE tap keyword found in %s"
          % os.path.basename(probe))

    if tap_names:
        tap = sorted(tap_names)[0]
        dst = os.path.join(tmp, 'tap_' + os.path.basename(probe))

        write_config(probe, dst, {tap: 0})
        got = read_config(dst).keywords[tap].value
        check("a_scalar_tap_stays_a_scalar",
              isinstance(got, (int, float)) and not isinstance(got, bool),
              "%s was written as the number 0 and read back as %r (%s). A tap whose "
              "value column holds a number is a fixed value, not a sweep."
              % (tap, got, type(got).__name__))

        rng = '[-0.34:.02:0]'
        write_config(probe, dst, {tap: rng})
        got = read_config(dst).keywords[tap].value
        check("a_range_string_tap_stays_a_string",
              isinstance(got, str) and got.strip() == rng,
              "%s was written as the sweep range %r and read back as %r (%s). "
              "Collapsing a range to a number silently shrinks the search space "
              "from ~198 points to 1 -- the exact confusion behind the TxFFE "
              "config-mismatch investigation."
              % (tap, rng, got, type(got).__name__))

        # and the engine must agree: the written range has to reach param as a
        # multi-point sweep, not a single value.
        pa, _ = _parse(dst)
        grid = [v for k, v in pa.items()
                if 'ffe' in k.lower() and np.asarray(v).size > 1]
        check("the_engine_sees_a_multi_point_sweep_after_a_range_is_written",
              grid,
              "%s was written as the sweep %r, but no Tx FFE field in param holds "
              "more than one point. The range reached the cell and did not reach "
              "the search space." % (tap, rng))

    # ------------------------------------------- 5. package blocks are distinct
    #
    # `.START <name>` / `.END` blocks each redefine A_v, C_p, R_d and friends.
    # The engine reads the main region and each block separately; a reader that
    # flattened the sheet would call those keywords duplicated (hiding them) and
    # a writer that flattened it would drop an edit into an arbitrary block.
    check("package_preset_blocks_were_parsed", len(doc.packages) > 1,
          "%s defines %d .START package block(s). The four-way repetition of "
          "A_v/C_p/R_d is the whole reason region-aware parsing exists, so a "
          "corpus without it cannot exercise this."
          % (os.path.basename(probe), len(doc.packages)))

    shared = set.intersection(*[set(v) for v in doc.packages.values()]) \
        if doc.packages else set()
    check("package_blocks_share_keyword_names", len(shared) >= 3,
          "package blocks share only %d keyword name(s) (%s). The hazard is a "
          "name defined in several blocks at once; without one, region-aware "
          "addressing is untested." % (len(shared), sorted(shared)[:5]))

    if shared:
        kw = sorted(shared)[0]
        blocks = sorted(doc.packages)
        coords = {b: doc.packages[b][kw].coord for b in blocks}
        check("the_same_keyword_in_different_blocks_has_different_cells",
              len(set(coords.values())) == len(blocks),
              "%s resolves to %s across blocks %s -- two blocks are pointing at "
              "one cell, so an edit to one would silently change the other"
              % (kw, coords, blocks))

        # write into exactly one block and prove the others did not move
        tgt = '%s/%s' % (blocks[0], kw)
        dst = os.path.join(tmp, 'pkg_' + os.path.basename(probe))
        write_config(probe, dst, {tgt: 0.4321})
        after = read_config(dst)
        moved = [b for b in blocks[1:]
                 if after.packages[b][kw].value != doc.packages[b][kw].value]
        check("editing_one_package_block_leaves_the_others_alone",
              not moved,
              "writing %s also changed %s in block(s) %s"
              % (tgt, kw, moved))
        check("editing_one_package_block_actually_changed_it",
              after.packages[blocks[0]][kw].value == 0.4321,
              "wrote 0.4321 to %s and read back %r"
              % (tgt, after.packages[blocks[0]][kw].value))

    # ------------------------------------------ 6. [TX RX] split and rejoin
    #
    # The Tx and Rx blocks each edit one half of a single cell, so the two
    # halves are recombined before writing. Recombining reformats the literal
    # (`[0.4e-4  0.9e-4 ;0.4e-4  0.9e-4 ]` comes back as
    # `[0.4e-4 0.9e-4 ; 0.4e-4 0.9e-4]`), and reformatting a value the engine
    # parses is exactly the kind of "harmless" change that is not harmless.
    # So: split every pair, put it straight back together, and require the
    # ENGINE to see no difference at all.
    from gui.config_io import split_tx_rx, join_tx_rx, is_tx_rx  # noqa: E402

    pairs = {k: v for k, v in doc.keywords.items() if is_tx_rx(v)}
    check("the_corpus_contains_tx_rx_paired_cells", pairs,
          "no cell in %s is marked [TX RX], so the split/rejoin path is "
          "unexercised" % os.path.basename(probe))

    if pairs:
        rebuilt = {}
        for k, v in pairs.items():
            tx, rx = split_tx_rx(v.value)
            rebuilt[k] = join_tx_rx(tx, rx, v.value)
        dst = os.path.join(tmp, 'txrx_' + os.path.basename(probe))
        write_config(probe, dst, rebuilt)

        pa, oa = _parse(probe)
        pb, ob = _parse(dst)
        bad = _diff(pa, pb, 'param') + _diff(oa, ob, 'OP')
        check("splitting_and_rejoining_a_tx_rx_cell_changes_nothing",
              not bad,
              "recombining %s without changing either half made the engine "
              "parse the config differently:\n     %s"
              % (sorted(pairs), "\n     ".join(bad[:10])))

        # and a real half-edit must land on the correct half
        k = sorted(pairs)[0]
        tx, rx = split_tx_rx(doc.keywords[k].value)
        dst2 = os.path.join(tmp, 'half_' + os.path.basename(probe))
        write_config(probe, dst2,
                     {k: join_tx_rx(tx, '0.123', doc.keywords[k].value)})
        got = split_tx_rx(read_config(dst2).keywords[k].value)
        check("editing_the_rx_half_leaves_the_tx_half_alone",
              got is not None and got[0] == tx and got[1] == '0.123',
              "set the RX half of %s to 0.123 and read back TX=%r RX=%r "
              "(TX was %r)" % (k, got and got[0], got and got[1], tx))

    print("\n%d configuration workbook(s) round-tripped through the engine."
          % len(configs))
    print("   [TX RX] paired cells: %s" % (sorted(pairs) or 'none'))
    print("   package blocks in %s: %s"
          % (os.path.basename(probe), sorted(doc.packages) or 'none'))
    print("   formula-backed keywords in %s: %s"
          % (os.path.basename(probe), formula_kws or 'none'))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

finish()
