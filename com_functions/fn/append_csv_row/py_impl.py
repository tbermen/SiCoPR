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
import numpy as np


def append_csv_row(file_path, header_cells, row_cells):
    """Append a CSV row (MATLAB lines 5157-5187).

    file_path:    target CSV path (opened in append mode).
    header_cells: iterable of header names, written once when the file is new.
    row_cells:    iterable of cell values, or empty/None to write only the header.
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as fid:
        if not file_exists:
            fid.write(','.join(str(h) for h in header_cells) + '\n')
        if row_cells is not None and len(row_cells) > 0:
            out = []
            for v in row_cells:
                if isinstance(v, (bool, np.bool_)):
                    out.append(f'{float(v):.6g}')          # numeric in MATLAB
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    out.append(f'{v:.6g}')
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
