import numpy as np


def OptFom_Itick_BoxSearch(itickn, cluster, BEST, box_mid, box_size):
    """Determine next itick for box search strategy (MATLAB lines 3580-3605).

    itickn: 1-based iteration counter (MATLAB convention kept for caller compatibility)
    cluster: array of box-midpoint itick values
    Returns (itick, BEST, skip_it).
    """
    cluster = np.asarray(cluster)
    CL = len(cluster)

    if itickn <= CL:
        itick = int(cluster[itickn - 1])  # 1-based itickn → 0-based index
        skip_it = 0
    else:
        if itickn == CL + 1:
            box_begin = int(BEST.itick_in_cluster) - box_mid
            box_end = box_begin + int(box_size) - 1
            full_box = np.arange(box_begin, box_end + 1)
            BEST.cluster = np.setdiff1d(full_box, [BEST.itick_in_cluster])

        BEST_cluster = np.asarray(BEST.cluster)
        if len(BEST_cluster) == 0:
            return [], BEST, 1

        pos = itickn - CL - 1  # 0-based position within BEST.cluster
        itick = int(BEST_cluster[pos])
        skip_it = 0

    return itick, BEST, skip_it
