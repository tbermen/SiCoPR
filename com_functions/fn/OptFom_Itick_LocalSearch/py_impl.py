import math


def OptFom_Itick_LocalSearch(itick, middle_search, BEST, LocalSearch_Value):
    """Return skip_it flag for itick local search (MATLAB lines 3606-3617)."""
    skip_it = 0
    if middle_search and LocalSearch_Value > 0:
        if (itick >= 0
                and not math.isinf(BEST.positive_itick_FOM)
                and abs(BEST.positive_itick_in_loop - itick) >= LocalSearch_Value):
            skip_it = 1
        if (itick <= 0
                and not math.isinf(BEST.negative_itick_FOM)
                and abs(BEST.negative_itick_in_loop - itick) >= LocalSearch_Value):
            skip_it = 1
    return skip_it
