# ============================================================
# MATLAB→Python translation notes for Full_Grid_Matrix
# MATLAB lines: 2119–2179
# ============================================================
# Creates full Cartesian product grid from a list of sweep variable arrays.
# MATLAB repmat/reshape use column-major order; equivalent is np.repeat + np.tile.
# repmat(col, [1, N]) where col is n×1 → B is n×N
# reshape(B', [n*N, 1]) in column-major = each element of col repeated N times.
# → Equivalent to np.repeat(col, N).
# Mixed cell/numeric inputs: all treated as Python lists here.
# ============================================================

import numpy as np


def Full_Grid_Matrix(in_list):
    if not isinstance(in_list, (list, tuple)):
        raise ValueError('input must be list of individual sweep variables')
    num_columns = len(in_list)
    num_cases = 1
    for col in in_list:
        num_cases *= len(col)

    out = [[None] * num_columns for _ in range(num_cases)]

    num_repetitions = 1
    for k in range(num_columns - 1, -1, -1):
        col = list(in_list[k])
        n = len(col)
        # Each element repeated num_repetitions times (MATLAB repmat+reshape column-major)
        C = []
        for elem in col:
            C.extend([elem] * num_repetitions)
        num_repeats = num_cases // len(C)
        D = C * num_repeats
        for row in range(num_cases):
            out[row][k] = D[row]
        num_repetitions *= n

    return out
