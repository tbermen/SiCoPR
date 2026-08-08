import numpy as np


def cursor_sample_index(sbr, param, OP, peak_search_range):
    """Find cursor sample index via Muller-Mueller criterion (MATLAB lines 5405-5479).

    All indices are 0-based in Python (MATLAB uses 1-based).

    Parameters
    ----------
    sbr               : pulse response array
    param             : struct with samples_per_ui, ndfe, bmax
    OP                : struct with CDR ('MM' or 'Mod-MM')
    peak_search_range : 0-based index array for peak search window

    Returns
    -------
    cursor_i          : 0-based cursor index (None if no zero crossing)
    no_zero_crossing  : 0 or 1 flag
    sbr_peak_i        : 0-based peak index
    zxi               : 0-based zero-crossing index (scalar array, empty if none)
    """
    sbr = np.asarray(sbr, dtype=float).ravel()
    peak_search_range = np.asarray(peak_search_range, dtype=int).ravel()
    M = int(param.samples_per_ui)

    # Find peak in search window
    sub = sbr[peak_search_range]
    sbr_peak_tmp = int(np.argmax(sub))
    max_of_sbr = float(sub[sbr_peak_tmp])
    sbr_peak_i = sbr_peak_tmp + int(peak_search_range[0])

    no_zero_crossing = 0
    cursor_i = None

    # Find zero crossings in [search_start, sbr_peak_i]
    search_start = max(0, sbr_peak_i - 4 * M)
    segment = sbr[search_start:sbr_peak_i + 1]
    diff_signs = np.diff(np.sign(segment - 0.01 * max_of_sbr))
    local_idx = np.where(diff_signs >= 1)[0]
    zxi_arr = local_idx + search_start  # 0-based global indices

    if len(zxi_arr) == 0:
        no_zero_crossing = 1
        return None, no_zero_crossing, sbr_peak_i, np.array([], dtype=int)

    if len(zxi_arr) > 1:
        zxi_arr = zxi_arr[[-1]]  # keep only last zero crossing

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
    cursor_i = zxi + mm_cursor_offset  # 0-based

    return cursor_i, no_zero_crossing, sbr_peak_i, zxi_arr
