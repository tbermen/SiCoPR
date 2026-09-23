# ============================================================
# MATLAB→Python translation notes for OptFom_Build_TXFFE
# MATLAB lines: 2734–2816
# ============================================================
# fieldnames(param) → vars(param).keys() for SimpleNamespace
# regexp → re.search
# txffe_sweep_indices: 1-based MATLAB convention (used by OptFom_Local_Search)
# Full_Grid_Matrix inlined as _Full_Grid_Matrix
# precursor/postcursor indices are 1-based lists, converted to 0-based for numpy
# postcursor_indices range-expansion: MATLAB (first:last) → Python range(first, last+1)
# TXFFE_grid converted to float numpy array for matrix operations
# ============================================================

import re
import numpy as np
from com_functions.fn.Full_Grid_Matrix.py_impl import Full_Grid_Matrix as _Full_Grid_Matrix


def _tap_values(param, field):
    """MATLAB `param.(this_tx_field)`, as Full_Grid_Matrix's `in{k}(:)`.

    Full_Grid_Matrix takes num_cases from cellfun('length',in) -- the LONGEST
    dimension -- but then lays out every element of in{k}(:), so a sweep
    variable that is not a vector makes num_cases/length(C) fractional and
    repmat refuse it.  COM Octave, tx_ffe_cm1_values = [0 .1 .2; .3 .4 .5]:
    "conversion of 0.5 to int64_t value failed".  ravel() had flattened it to
    six values and answered.
    """
    val = np.asarray(getattr(param, field))
    if val.ndim > 1 and val.size != max(val.shape):
        raise ValueError('param.%s must be a vector, got shape %s'
                         % (field, val.shape))
    return list(val.ravel())


def OptFom_Build_TXFFE(param):
    param_fields = list(vars(param).keys())

    # MATLAB's regexp is not anchored, so a field that merely CONTAINS
    # tx_ffe_cm<n>_values is counted -- and the sprintf'd lookup that follows
    # then fails.  COM Octave, param.xtx_ffe_cm1_values_extra = [0 0.1]:
    # "structure has no member 'tx_ffe_cm1_values'".  The anchored ^...$ here
    # skipped such a field and answered.
    num_pre = sum(1 for f in param_fields if re.search(r'tx_ffe_cm\d+_values', f))
    num_post = sum(1 for f in param_fields if re.search(r'tx_ffe_cp\d+_values', f))
    num_taps = num_pre + num_post
    cur = num_pre + 1  # 1-based cursor position

    txffe_cell = [None] * num_taps
    for k in range(num_pre, 0, -1):  # k from num_pre down to 1
        idx = num_pre - k  # 0-based
        txffe_cell[idx] = _tap_values(param, f'tx_ffe_cm{k}_values')
    for k in range(1, num_post + 1):
        idx = k + num_pre - 1  # 0-based
        txffe_cell[idx] = _tap_values(param, f'tx_ffe_cp{k}_values')

    txffe_lengths = np.array([len(c) for c in txffe_cell], dtype=int)

    # 1-based indices of taps with more than one value, sorted by descending
    # length.  MATLAB's sort is STABLE, so ties keep ascending tap order;
    # argsort(ascending, stable)[::-1] reverses them.  COM Octave, tap lengths
    # [2 4 2 4], gives [2 4 1 3] where the reversed form gave [4 2 3 1] -- and
    # the plainest real case, one swept precursor and one swept postcursor of
    # equal length, came out [2 1] instead of [1 2].
    raw_sweep = np.where(txffe_lengths > 1)[0] + 1  # 1-based
    if len(raw_sweep) > 0:
        length_sort = np.argsort(-txffe_lengths[raw_sweep - 1], kind='stable')
        txffe_sweep_indices = raw_sweep[length_sort]
    else:
        txffe_sweep_indices = np.array([], dtype=int)

    if num_taps == 0:
        TXFFE_grid = np.array([[0.0]])
        FULL_tx_index_vector = [[1]]
    else:
        grid_raw = _Full_Grid_Matrix(txffe_cell)
        TXFFE_grid = np.array(grid_raw, dtype=float)
        txffe_index_cell = [list(range(1, int(txffe_lengths[k]) + 1))
                            for k in range(num_taps)]
        FULL_tx_index_vector = _Full_Grid_Matrix(txffe_index_cell)

    # Build precursor/postcursor indices (1-based) adjusting cur for leading zeros
    cur_start = cur  # original cursor position (1-based)
    precursor_indices = []
    postcursor_indices = []
    auto_count_trigger = False
    for kv in range(1, num_taps + 1):  # 1-based kv
        cell_vals = txffe_cell[kv - 1]
        if (not auto_count_trigger
                and len(cell_vals) == 1
                and float(cell_vals[0]) == 0.0):
            if kv < cur_start:
                cur -= 1
        else:
            if kv < cur_start:
                auto_count_trigger = True
                precursor_indices.append(kv)
            else:
                auto_count_trigger = False
                postcursor_indices.append(kv)

    if postcursor_indices:
        postcursor_indices = list(range(postcursor_indices[0], postcursor_indices[-1] + 1))

    txffe_cursor_vector = (1.0 - np.sum(np.abs(TXFFE_grid), axis=1, keepdims=True))

    pre_cols = [i - 1 for i in precursor_indices]  # 0-based
    post_cols = [i - 1 for i in postcursor_indices]  # 0-based

    parts = []
    if pre_cols:
        parts.append(TXFFE_grid[:, pre_cols])
    parts.append(txffe_cursor_vector)
    if post_cols:
        parts.append(TXFFE_grid[:, post_cols])
    txffe_matrix = np.hstack(parts)

    return txffe_matrix, cur, txffe_sweep_indices, np.array(FULL_tx_index_vector), txffe_cursor_vector
