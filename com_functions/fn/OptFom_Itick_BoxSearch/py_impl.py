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
        # COM Octave: cluster(itickn) is a subscript, so itickn below 1 is an
        # error, NOT a wrap to the end of the vector:
        #   itickn=0  -> error: cluster(0): subscripts must be either
        #                integers 1 to (2^63)-1 or logicals
        #   itickn=-1 -> error: cluster(-1): same
        if itickn < 1:
            raise IndexError('cluster(%s): subscripts must be either integers '
                             '1 to (2^63)-1 or logicals' % itickn)
        itick = cluster[itickn - 1]  # 1-based itickn → 0-based index
        skip_it = 0
    else:
        if itickn == CL + 1:
            # COM Octave: BEST.itick_in_cluster is [] until some itick beats
            # the starting FOM, and MATLAB propagates that emptiness --
            # [] - box_mid is [], the colon range is empty and setdiff of an
            # empty box is empty, so this is the 'every case was bad' path:
            #   BEST.itick_in_cluster=[] -> itick=[], skip_it=1, BEST.cluster=[]
            if np.size(BEST.itick_in_cluster) == 0:
                BEST.cluster = np.array([])
            else:
                # No int() on the box bounds: MATLAB carries the value through
                # untouched, so a non-integer midpoint gives a non-integer box.
                # COM Octave, itick_in_cluster=8.5, box_mid=2, box_size=5:
                #   BEST.cluster = [6.5 7.5 9.5 10.5], itick = 6.5
                box_begin = BEST.itick_in_cluster - box_mid
                box_end = box_begin + box_size - 1
                full_box = np.arange(box_begin, box_end + 1)
                BEST.cluster = np.setdiff1d(full_box, [BEST.itick_in_cluster])

        BEST_cluster = np.asarray(BEST.cluster)
        if len(BEST_cluster) == 0:
            return [], BEST, 1

        pos = itickn - CL - 1  # 0-based position within BEST.cluster
        itick = BEST_cluster[pos]
        skip_it = 0

    return itick, BEST, skip_it
