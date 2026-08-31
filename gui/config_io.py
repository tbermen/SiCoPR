"""Read and write COM configuration workbooks without damaging them.

Phase 0 of the schematic GUI. No UI here: this is the part everything else
stands on, and it is where the risk actually lives.

WHAT THE ENGINE ACTUALLY REQUIRES
---------------------------------
`read_ParamConfigFile` loads the workbook with `data_only=True` and, for each
keyword, scans a region of the `COM_Settings` grid for a cell whose stripped,
lower-cased text equals the keyword, then takes **the cell immediately to its
right** as the value. The keyword must appear exactly once in that region; two
matches is an error. Position is otherwise irrelevant.

The region matters. Column A can carry `.START <name>` / `.END` markers that
delimit per-package preset blocks; the engine truncates the main parameter
region to everything BEFORE the first `.START` and parses each block separately.
A stock config has four package blocks, so `A_v`, `C_p` and `R_d` each appear
four times without ever being ambiguous to the engine. A reader that treats the
sheet as one flat table gets this wrong in both directions -- it hides real
settings as "duplicated", and it would let an edit land in an arbitrary package
block.

WHY THIS DOES NOT USE openpyxl TO SAVE
--------------------------------------
The stock configs contain 22 formulas in `COM_Settings`, nine of which are value
cells for real keywords: `f_v`, `f_f`, `f_n`, `TR_TDR`, `N_tail_start`, `f_z`,
`f_p1`, `f_p2` and `f_HP_PZ`. openpyxl can load formulas OR cached values, never
both, and it does not evaluate:

  * load(data_only=False) + save  ->  formula kept, CACHED VALUE LOST. The
    engine reads cached values, so `f_v` would come back as None. Silently
    broken config.
  * load(data_only=True) + save   ->  every formula in the book flattened to a
    literal. The engine is happy, but the workbook stops tracking `f_b` and a
    human editing it later gets no warning.

Neither is acceptable, so writes are made directly to the sheet XML inside the
xlsx zip: only the targeted cells change and every other byte is preserved,
formulas and cached values included.

THE STALE-DERIVED-VALUE HAZARD
------------------------------
Because the engine reads cached values and nothing here evaluates formulas,
editing a cell that a formula depends on leaves the dependent cached value
stale, and the engine will read the stale one. `write_config` detects this and
returns it as a warning rather than silently producing a config whose numbers
disagree with its own arithmetic.

    from gui.config_io import read_config, write_config
    doc = read_config('config.xlsx')
    doc.keywords['f_b'].value                 # 106.25
    warnings = write_config('config.xlsx', 'new.xlsx', {'f_b': 112.5})
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import shutil
import zipfile
from types import SimpleNamespace

import openpyxl

SHEET = 'COM_Settings'


class KeywordCell(SimpleNamespace):
    """One config keyword: where it lives, what it holds, how it is written."""


def _col_letters(idx0):
    """0-based column index -> spreadsheet letters."""
    s, n = '', idx0 + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _region_keywords(grid, ws_fml, lo, hi):
    """Keyword cells in grid[lo:hi], addressed the way `_xls_param` addresses
    them: label anywhere in the region, value in the cell to its right, and the
    label unique WITHIN the region (a second occurrence makes the engine raise).
    """
    seen = {}
    for r in range(lo, hi):
        for c, cell in enumerate(grid[r]):
            if isinstance(cell, str) and cell.strip():
                seen.setdefault(cell.strip().lower(), []).append((r, c))

    out = {}
    for r in range(lo, hi):
        row = grid[r]
        for c, cell in enumerate(row):
            if not (isinstance(cell, str) and cell.strip()):
                continue
            label = cell.strip()
            if len(seen[label.lower()]) != 1:
                continue                      # ambiguous: the engine would raise
            if c + 1 >= len(row) or row[c + 1] is None:
                continue                      # a heading or a units cell
            coord = '%s%d' % (_col_letters(c + 1), r + 1)
            fml = ws_fml[coord].value
            is_f = isinstance(fml, str) and fml.startswith('=')
            out[label] = KeywordCell(
                name=label, value=row[c + 1],
                units=row[c + 2] if c + 2 < len(row) else None,
                info=row[c + 3] if c + 3 < len(row) else None,
                row=r, col=c, coord=coord,
                is_formula=is_f, formula=fml if is_f else None)
    return out


def read_config(path):
    """Parse a configuration workbook the way the engine parses it.

    The sheet is not one flat table. Column A may carry `.START <name>` /
    `.END` markers delimiting per-package preset blocks, and
    `read_ParamConfigFile` truncates the main parameter region to everything
    BEFORE the first `.START`, then parses each block on its own. That is why
    `A_v`, `C_p`, `R_d` and friends appear four times in a stock config and the
    engine still does not trip its own duplicate-keyword error: each occurrence
    lives in a different package block.

    Reading the sheet as one region would make a keyword look ambiguous when it
    is not, and -- worse for a writer -- would let an edit land in whichever
    package block happened to match first.

    Returns a namespace with:
      keywords   main-region settings, name -> KeywordCell
      packages   {block name: {name -> KeywordCell}} from the .START blocks
      schema     {keyword: {matlab, default, info}} from the keywords_* sheet
      grid, sheet, sheetnames, path
    """
    wb_val = openpyxl.load_workbook(path, data_only=True)
    wb_fml = openpyxl.load_workbook(path, data_only=False)
    name = SHEET if SHEET in wb_val.sheetnames else wb_val.sheetnames[0]
    ws_fml = wb_fml[name]

    grid = [list(r) for r in wb_val[name].iter_rows(values_only=True)]

    col_a = [(row[0] if row else None) for row in grid]
    starts = [i for i, v in enumerate(col_a) if v == '.START']
    ends = [i for i, v in enumerate(col_a) if v == '.END']

    packages = {}
    main_hi = len(grid)
    if starts:
        if len(starts) != len(ends):
            raise ValueError('%s: %d .START markers but %d .END'
                             % (os.path.basename(path), len(starts), len(ends)))
        main_hi = starts[0]
        for s, e in zip(starts, ends):
            pkg_name = str(grid[s][1])
            packages[pkg_name] = _region_keywords(grid, ws_fml, s + 1, e)

    keywords = _region_keywords(grid, ws_fml, 0, main_hi)

    return SimpleNamespace(path=path, sheet=name, grid=grid, keywords=keywords,
                           packages=packages, schema=_read_schema(wb_val),
                           main_rows=(0, main_hi),
                           sheetnames=list(wb_val.sheetnames))


def _read_schema(wb):
    """The `keywords_*` sheet: the config's own documentation of itself.

    Columns are `matlab variable | keyword | default | info`. This is what makes
    a GUI able to show a real description and a real default instead of guessing
    from the name.
    """
    sheets = [s for s in wb.sheetnames if s.lower().startswith('keywords')]
    if not sheets:
        return {}
    out = {}
    for row in wb[sheets[0]].iter_rows(values_only=True):
        if not row or len(row) < 2 or not row[1]:
            continue
        kw = str(row[1]).strip()
        if kw.lower() == 'keyword':
            continue
        out[kw] = {
            'matlab': str(row[0]).strip() if row[0] else '',
            'default': row[2] if len(row) > 2 else None,
            'info': str(row[3]).strip() if len(row) > 3 and row[3] else '',
        }
    return out


def _rows_of(value):
    """A MATLAB-ish literal split into rows of tokens.

    `'[0.3e-4  0.3e-4 ]'`            -> [['0.3e-4', '0.3e-4']]
    `'[0.13 0.15 ; 0.13 0.15 ]'`     -> [['0.13','0.15'], ['0.13','0.15']]
    `'PKG_A PKG_B'`                  -> [['PKG_A', 'PKG_B']]
    """
    s = str(value).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    rows = []
    for part in s.split(';'):
        toks = [t for t in re.split(r'[,\s]+', part.strip()) if t]
        rows.append(toks)
    return rows


def split_tx_rx(value):
    """(tx_text, rx_text) for a value the sheet marks `[TX RX]`, else None.

    Several die and package settings are one cell holding both ends of the
    link. The engine wants the combined vector, but a person editing "the Rx
    die" should not have to hand-edit the second half of a matrix that lives
    under a Tx heading -- which is what made the Rx blocks look almost empty.

    Two shapes occur: two rows (row 1 = TX, row 2 = RX), or one row of exactly
    two elements. Anything else is not splittable and returns None rather than
    being guessed at.
    """
    rows = _rows_of(value)
    if len(rows) == 2 and all(rows):
        return ' '.join(rows[0]), ' '.join(rows[1])
    if len(rows) == 1 and len(rows[0]) == 2:
        return rows[0][0], rows[0][1]
    return None


def join_tx_rx(tx_text, rx_text, original):
    """Recombine the two halves in the shape `original` had.

    Writing back a different shape than the engine expects is the whole risk
    here, so the original literal decides the format, not the new text.
    """
    orig = str(original).strip()
    bracketed = orig.startswith('[') and orig.endswith(']')
    rows = _rows_of(original)
    tx, rx = str(tx_text).strip(), str(rx_text).strip()
    if len(rows) == 2:
        body = '%s ; %s' % (tx, rx)
    else:
        body = '%s %s' % (tx, rx)
    return '[%s]' % body if bracketed else body


def is_tx_rx(cell):
    """Does the sheet itself mark this cell as carrying both ends?

    Driven by the workbook's own annotation column (`[TX RX]`) rather than a
    hand-kept list here, so a config that adds or drops one stays correct.
    """
    note = str(getattr(cell, 'info', '') or '')
    if not ('tx' in note.lower() and 'rx' in note.lower()):
        return False
    return split_tx_rx(cell.value) is not None


_ENGINE_KEYWORDS = None


def engine_keywords():
    """Every keyword `sicopr.py` actually looks up, read out of the source.

    A config sheet contains more label/value pairs than it has settings: units
    columns, range templates like `[min:step:max]`, and reference values a human
    typed in for context (`T_t`, `T_ft`, `T_nt` are in the stock configs and are
    read by neither the port nor the MATLAB original). Those are indistinguishable
    from settings by shape alone -- each is a string with something to its right.

    The engine's own lookups are the authoritative list, so they are what gets
    used, rather than a hand-maintained set here that would drift.
    """
    global _ENGINE_KEYWORDS
    if _ENGINE_KEYWORDS is None:
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'sicopr.py')
        with open(src, encoding='utf-8') as f:
            text = f.read()
        pat = re.compile(r"__xls_param\(\s*(?:parameter|block)\s*,\s*"
                         r"(['\"])(.+?)\1")
        _ENGINE_KEYWORDS = {m.group(2) for m in pat.finditer(text)}
    return _ENGINE_KEYWORDS


def is_setting(name):
    """Is this label a setting the engine reads, or sheet decoration?"""
    return name.strip().lower() in {k.lower() for k in engine_keywords()}


def all_cells(doc):
    """Every addressable keyword, main region and package blocks alike, keyed
    the way `write_config` accepts them: `name`, or `BLOCK/name`."""
    out = dict(doc.keywords)
    for pkg, cells in doc.packages.items():
        for k, v in cells.items():
            out['%s/%s' % (pkg, k)] = v
    return out


def _formula_precedents(doc):
    """coord -> set of coords its formula references. Crude but sufficient:
    the config formulas are simple arithmetic over single cells."""
    out = {}
    for kw in all_cells(doc).values():
        if not kw.is_formula:
            continue
        refs = set(re.findall(r'\$?([A-Z]{1,3})\$?(\d+)', kw.formula))
        out[kw.coord] = {'%s%s' % (a, b) for a, b in refs}
    return out


def _sheet_xml_name(path, sheet_name):
    """Which xl/worksheets/sheetN.xml holds `sheet_name`."""
    with zipfile.ZipFile(path) as z:
        wb = z.read('xl/workbook.xml').decode('utf-8')
        rels = z.read('xl/_rels/workbook.xml.rels').decode('utf-8')
    m = re.search(r'<sheet[^>]*name="%s"[^>]*r:id="([^"]+)"' % re.escape(sheet_name), wb)
    if not m:
        m = re.search(r'<sheet[^>]*r:id="([^"]+)"', wb)
    rid = m.group(1)
    m2 = re.search(r'<Relationship[^>]*Id="%s"[^>]*Target="([^"]+)"' % re.escape(rid), rels)
    target = m2.group(1)
    if target.startswith('/'):
        return target.lstrip('/')
    return 'xl/' + target.lstrip('./')


def _fmt(value):
    """(xml_attrs, inner_xml) for a cell value."""
    if isinstance(value, bool):
        return ' t="b"', '<v>%d</v>' % int(value)
    if isinstance(value, (int, float)):
        return '', '<v>%s</v>' % repr(value)
    text = str(value)
    for a, b in (('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;')):
        text = text.replace(a, b)
    # inlineStr avoids having to touch the shared-strings table at all
    return ' t="inlineStr"', '<is><t xml:space="preserve">%s</t></is>' % text


def write_config(src, dst, changes):
    """Copy `src` to `dst`, setting each keyword in `changes` to its new value.

    Only the targeted cells are rewritten; every other byte of the workbook is
    preserved, including formulas and their cached values.

    Returns a list of warning strings. An empty list means a clean write.
    Raises KeyError for an unknown or ambiguous keyword -- better than writing a
    config that silently ignores half of what was asked for.
    """
    doc = read_config(src)
    cells = all_cells(doc)
    warnings = []

    unknown = [k for k in changes if k not in cells]
    if unknown:
        raise KeyError('not addressable keywords in %s: %s'
                       % (os.path.basename(src), sorted(unknown)))

    # Overwriting a formula cell replaces it with a literal. That is the correct
    # reading of "the user set this value", but it must be said out loud.
    for k in changes:
        if cells[k].is_formula:
            warnings.append(
                '%s was a formula (%s); it is now the literal value %r'
                % (k, cells[k].formula, changes[k]))

    # Editing a precedent leaves dependent cached values stale, and the engine
    # reads cached values. Report it rather than emit a self-inconsistent book.
    changed_coords = {cells[k].coord for k in changes}
    for coord, refs in _formula_precedents(doc).items():
        if coord in changed_coords:
            continue
        hit = refs & changed_coords
        if hit:
            owner = next((k for k, kw in cells.items()
                          if kw.coord == coord), coord)
            warnings.append(
                'STALE: %s is computed by a formula referencing %s, which this '
                'write changes. Nothing here evaluates formulas and the engine '
                'reads cached values, so %s will keep its old number until the '
                'workbook is opened in a spreadsheet and recalculated.'
                % (owner, ', '.join(sorted(hit)), owner))

    edits = {cells[k].coord: v for k, v in changes.items()}
    _rewrite_cells(src, dst, doc.sheet, edits)
    return warnings


def _rewrite_cells(src, dst, sheet_name, edits):
    """Copy the xlsx, replacing only the given cells in the given sheet."""
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    target = _sheet_xml_name(src, sheet_name)

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    xml = blobs[target].decode('utf-8')
    for coord, value in edits.items():
        attrs, inner = _fmt(value)
        # keep the style index so the cell still looks like its neighbours
        m = re.search(r'<c r="%s"([^>]*?)(/>|>.*?</c>)' % re.escape(coord), xml,
                      flags=re.S)
        style = ''
        if m:
            sm = re.search(r'\ss="(\d+)"', m.group(1))
            if sm:
                style = ' s="%s"' % sm.group(1)
            xml = xml[:m.start()] + '<c r="%s"%s%s>%s</c>' % (coord, style, attrs, inner) \
                + xml[m.end():]
        else:
            # cell absent from the sheet XML: insert it in row order
            rown = int(re.match(r'[A-Z]+(\d+)', coord).group(1))
            rm = re.search(r'<row[^>]*r="%d"[^>]*>' % rown, xml)
            if not rm:
                raise KeyError('cannot place %s: row %d not in sheet XML'
                               % (coord, rown))
            ins = '<c r="%s"%s>%s</c>' % (coord, attrs, inner)
            xml = xml[:rm.end()] + ins + xml[rm.end():]
    blobs[target] = xml.encode('utf-8')

    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
