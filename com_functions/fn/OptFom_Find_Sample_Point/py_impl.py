import numpy as np


def _cursor_sample_index(sbr, param, OP, peak_search_range):
    """Inline copy of cursor_sample_index (MATLAB lines 5405-5479)."""
    sbr = np.asarray(sbr, dtype=float).ravel()
    peak_search_range = np.asarray(peak_search_range, dtype=int).ravel()
    M = int(param.samples_per_ui)

    sub = sbr[peak_search_range]
    sbr_peak_tmp = int(np.argmax(sub))
    max_of_sbr = float(sub[sbr_peak_tmp])
    sbr_peak_i = sbr_peak_tmp + int(peak_search_range[0])

    no_zero_crossing = 0
    cursor_i = None

    search_start = max(0, sbr_peak_i - 4 * M)
    segment = sbr[search_start:sbr_peak_i + 1]
    diff_signs = np.diff(np.sign(segment - 0.01 * max_of_sbr))
    local_idx = np.where(diff_signs >= 1)[0]
    zxi_arr = local_idx + search_start

    if len(zxi_arr) == 0:
        no_zero_crossing = 1
        return None, no_zero_crossing, sbr_peak_i, np.array([], dtype=int)

    if len(zxi_arr) > 1:
        zxi_arr = zxi_arr[[-1]]

    zxi = int(zxi_arr[0])
    max_dfe1 = 0.0 if param.ndfe == 0 else float(np.asarray(param.bmax).ravel()[0])

    mm_range = np.arange(zxi, zxi + 2 * M + 1, dtype=int)
    cdr = str(OP.CDR).strip()
    if cdr == 'Mod-MM':
        mm_metric = np.abs(sbr[mm_range + M] - max_dfe1 * sbr[mm_range])
    else:
        mm_metric = np.abs(
            sbr[mm_range - M] - np.maximum(sbr[mm_range + M] - max_dfe1 * sbr[mm_range], 0.0)
        )

    mm_cursor_offset = int(np.argmin(mm_metric))
    cursor_i = zxi + mm_cursor_offset
    return cursor_i, no_zero_crossing, sbr_peak_i, zxi_arr


def OptFom_Find_Sample_Point(sbr, param, OP, search_range):
    """Find cursor sample point with optional ts_anchor override (MATLAB lines 3515-3537).

    ts_anchor=0: use Muller-Mueller result as-is.
    ts_anchor=1: use peak sample directly.
    ts_anchor=2: use max-DV (max of cursor minus precursor window).
    All indices are 0-based.
    """
    sbr = np.asarray(sbr, dtype=float).ravel()
    cursor_i, no_zero_crossing, sbr_peak_i, _ = _cursor_sample_index(sbr, param, OP, search_range)

    ts_anchor = int(param.ts_anchor)
    M = int(param.samples_per_ui)

    if ts_anchor == 0:
        pass  # keep MM result unchanged
    elif ts_anchor == 1:
        cursor_i = sbr_peak_i
        no_zero_crossing = 0
    elif ts_anchor == 2:
        possible_cursor = sbr[sbr_peak_i - M:sbr_peak_i + M + 1]
        possible_precursor = sbr[sbr_peak_i - 2 * M:sbr_peak_i + 1]
        d_idx = int(np.argmax(possible_cursor - possible_precursor))
        cursor_i = sbr_peak_i - M + d_idx  # 0-based
        no_zero_crossing = 0
    else:
        raise ValueError('ts_anchor parameter must be 0, 1, or 2')

    return cursor_i, no_zero_crossing, sbr_peak_i
