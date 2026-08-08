import math
import numpy as np


def OptFom_Setup_Sampler_Sweep(full_sample_range, BEST, OP):
    """Initialize BEST itick tracking and compute loop_range for sampler sweep (MATLAB lines 3804-3842).

    Returns (loop_range, BEST, middle_search, box_search, cluster, box_mid).
    loop_range contains 0-based indices into full_sample_range (or iteration counters for box_search).
    """
    full_sample_range = np.asarray(full_sample_range)

    BEST.positive_itick_FOM = -math.inf
    BEST.negative_itick_FOM = -math.inf
    BEST.positive_itick_in_loop = []
    BEST.negative_itick_in_loop = []
    BEST.itick_FOM = -math.inf
    BEST.itick_in_cluster = []
    BEST.cluster = []

    box_size = int(OP.itick_box_size)
    cluster = np.array([])
    box_mid = []

    si = np.argsort(np.abs(full_sample_range))  # sort indices by |value|

    mode = str(OP.TS_SRCH_MODE).strip().lower()
    if mode == 'full-sweep':
        box_search = 0
        middle_search = 0
    elif mode == 'middle':
        box_search = 0
        middle_search = 1
    else:
        raise ValueError(f'Unsupported TS_SRCH_MODE: {OP.TS_SRCH_MODE!r}')

    if box_search:
        box_mid = int(box_size // 2)
        cluster = np.arange(full_sample_range[0] + box_mid,
                            full_sample_range[-1] + 1, box_size)
        CL = len(cluster)
        loop_range = np.arange(1, CL + box_size)  # 1-based for itickn arg of OptFom_Itick_BoxSearch
    elif middle_search:
        loop_range = si
    else:
        loop_range = np.arange(len(full_sample_range))

    return loop_range, BEST, middle_search, box_search, cluster, box_mid
