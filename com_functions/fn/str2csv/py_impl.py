# ============================================================
# MATLAB→Python translation notes for str2csv
# MATLAB lines: 11284–11291
# ============================================================
# 1-based vs 0-based indexing: not applicable (string ops only).
# Column-major expansion: cell_tmp{:} on a 2×n cell array expands
#   column-by-column: col1_row1, col1_row2, col2_row1, col2_row2, ...
#   i.e. interleaved: c{1}, ',', c{2}, ',', ..., c{n}, ''
#   This is exactly what str.join does.
# strcat on cell arrays: no trailing-space trimming (unlike char arrays).
# Output shape: scalar string.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================


def str2csv(c):
    """Join a list of strings with commas (MATLAB cell-array to CSV string).

    Equivalent to MATLAB str2csv — no trailing comma.
    Empty input returns ''.
    """
    return ','.join(c)


if __name__ == "__main__":
    print(repr(str2csv(['a', 'b', 'c'])))      # 'a,b,c'
    print(repr(str2csv(['hello', 'world'])))    # 'hello,world'
    print(repr(str2csv(['single'])))            # 'single'
    print(repr(str2csv([])))                    # ''
