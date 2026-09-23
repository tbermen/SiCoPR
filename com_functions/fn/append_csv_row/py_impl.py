# ============================================================
# MATLAB→Python translation notes for append_csv_row
# MATLAB lines: 5157-5187 (com_ieee8023_4p15p0_adaptive_local_search.m)
# NEW in Hansel D'silva's adaptive-local-search branch.
# ============================================================
# Appends one CSV row to file_path, writing the header first if the file
# does not yet exist. Numeric cells -> '%.6g'; string cells -> quoted;
# anything else -> '""'. This is the per-candidate trajectory logger used
# (optionally) by OptFom_Adaptive_Local_Search and is the data source for
# the sweep-vs-adaptive comparison visualisation.
# ============================================================

import os
import warnings

import numpy as np


def _is_numeric(v):
    """MATLAB isnumeric(): true for double/single/int*/complex, FALSE for
    logical.  A logical therefore falls through to the else branch and is
    written as "".  COM Octave, append_csv_row(f,{'h'},{true}) writes
    h\\n""\\n, not h\\n1\\n; bool is an int subclass in Python and the port
    wrote 1.
    """
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, complex, np.number)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind in 'iufc'      # 'b' is logical, which is not
    return False


def _sprintf_g6(v):
    """MATLAB `sprintf('%.6g', v)`.

    COM Octave, one cell at a time:
      {Inf} -> "Inf",  {-Inf} -> "-Inf",  {NaN} -> "NaN"   (Python: inf/nan)
      {[]}  -> ""       an empty numeric formats to an empty field, not ""
      {[1 2 3]}    -> "123"    the format is reapplied per element, no
      {[1 2; 3 4]} -> "1324"   separator, in COLUMN-MAJOR order
      {1+2i}       -> "1"      the imaginary part is dropped
    The port answered "" for every one of the array cases and lower-case
    inf/nan for the others.
    """
    a = np.asarray(v)
    if np.iscomplexobj(a):
        a = a.real
    out = []
    for x in np.asarray(a, dtype=float).ravel(order='F'):
        if np.isnan(x):
            out.append('NaN')
        elif np.isinf(x):
            out.append('Inf' if x > 0 else '-Inf')
        else:
            out.append('%.6g' % x)
    return ''.join(out)


def append_csv_row(file_path, header_cells, row_cells):
    """Append a CSV row (MATLAB lines 5157-5187).

    file_path:    target CSV path (opened in append mode).
    header_cells: iterable of header names, written once when the file is new.
    row_cells:    iterable of cell values, or empty/None to write only the header.
    """
    file_exists = os.path.isfile(file_path)
    try:
        fid = open(file_path, 'a', newline='')
    except OSError:
        # MATLAB: `if fid == -1, warning('Could not open %s', ...); return;`
        # COM Octave with a path under a directory that does not exist returns
        # normally after warning; the port raised FileNotFoundError.
        warnings.warn('Could not open %s' % file_path)
        return
    with fid:
        if not file_exists:
            # strjoin(header_cells, ',') takes a cell array of strings and
            # errors on anything else.  COM Octave, header {1,'b'}:
            # "error: Invalid call to strjoin."
            bad = [h for h in header_cells if not isinstance(h, str)]
            if bad:
                raise TypeError(
                    'append_csv_row: strjoin needs a cell array of strings; '
                    'header_cells contains %r' % (bad[0],))
            fid.write(','.join(header_cells) + '\n')
        if row_cells is not None and len(row_cells) > 0:
            out = []
            for v in row_cells:
                if _is_numeric(v):
                    out.append(_sprintf_g6(v))
                elif isinstance(v, str):
                    out.append(f'"{v}"')                   # quote strings for CSV safety
                else:
                    out.append('""')
            fid.write(','.join(out) + '\n')


if __name__ == '__main__':
    import tempfile
    p = os.path.join(tempfile.gettempdir(), 'append_csv_row_smoke.csv')
    if os.path.isfile(p):
        os.remove(p)
    append_csv_row(p, ['a', 'b', 'reason'], [])
    append_csv_row(p, ['a', 'b', 'reason'], [1, 2.5, 'hello'])
    print(open(p).read())
