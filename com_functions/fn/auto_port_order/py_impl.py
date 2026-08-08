# ============================================================
# MATLAB→Python translation notes for auto_port_order
# MATLAB lines: 4932-5057 (com_ieee8023_4p15p0.m) — NEW in r4p15p0
# ============================================================
# Automatically determines the 4-port reorder vector from the
# S-parameter cube by inspecting the lowest-frequency magnitude
# matrix (through energy) and a phase-delay near/far heuristic.
#
# 1-based vs 0-based: MATLAB returns 1-based port indices used as
#   sch(:,port_order,port_order). The rest of this Python codebase
#   keeps port_order 1-based and subtracts 1 at indexing time
#   (see read_Nport_touchstone), so this function ALSO returns
#   1-based indices (a Python list of ints) to match that convention.
# sch indexing: MATLAB sch(:,a,b) -> Python sch[:, a-1, b-1].
# round(): MATLAB rounds half away from zero; we replicate with
#   _mround so the quarter/three-quarter slice matches exactly.
# Output: list[int] length 4, 1-based.
# ============================================================

import numpy as np


def _mround(x):
    """MATLAB round(): half away from zero."""
    return int(np.floor(float(x) + 0.5)) if x >= 0 else int(np.ceil(float(x) - 0.5))


def auto_port_order(sch, F, flip_victim=0):
    """Automatically determine 4-port order (MATLAB lines 4932-5057).

    sch: (nfreq, nport, nport) complex S-parameter cube.
    F:   (nfreq,) frequency axis [Hz].
    flip_victim: if truthy, swap which pair is the Rx side.

    Returns port_order: list of 4 ints, 1-based (e.g. [1, 3, 2, 4]).
    """
    sch = np.asarray(sch)
    F = np.asarray(F, dtype=float).ravel()

    MinThruEnergy = 0.1

    num_ports = sch.shape[2]
    if num_ports != 4:
        raise ValueError('Auto Port Order routine only works for 4 port S-parameters')

    # Observe 1st frequency point
    LowFreq_Matrix = np.abs(sch[0, :, :])
    Raw_LowFreq_Matrix = LowFreq_Matrix.copy()

    # Ignore RL: set diagonal terms to 0
    LowFreq_Matrix = LowFreq_Matrix - np.diag(np.diag(LowFreq_Matrix))

    # Force reciprocity by taking the max element of any reciprocal term
    upper_triangle = np.triu(LowFreq_Matrix)
    lower_triangle = np.tril(LowFreq_Matrix).T
    max_matrix = np.maximum(upper_triangle, lower_triangle)
    LowFreq_Matrix = max_matrix + np.triu(max_matrix).T

    # Find connected ports by observing max value in each column (1-based indices)
    ConnectedPorts = np.zeros(4, dtype=int)
    for k in range(4):
        col = LowFreq_Matrix[:, k]
        idx = int(np.argmax(col))
        if col[idx] < MinThruEnergy:
            raise ValueError('Unable to determine port connections:  Low Energy')
        ConnectedPorts[k] = idx + 1  # 1-based

    # Force that connected ports agree with each other
    for k in range(1, 5):
        my_connection = ConnectedPorts[k - 1]
        other_connection = ConnectedPorts[my_connection - 1]
        if other_connection != k:
            raise ValueError('Unable to determine port connections:  Ambiguous connections')

    # Set initial port order (1-based, stored in a length-4 list)
    port_order = [0, 0, 0, 0]
    port_order[0] = 1
    port_order[2] = int(ConnectedPorts[0])           # 3rd index = connection to port 1
    others = sorted(set(range(1, 5)) - {1, int(ConnectedPorts[0])})
    port_order[1] = others[0]                          # 2nd index = min of remaining
    port_order[3] = others[-1]                         # 4th index = max of remaining

    if ConnectedPorts[port_order[1] - 1] != port_order[3]:
        raise ValueError('Unable to determine port connections:  Ambiguous connections')

    # Determine if port_order([2 4]) should be swapped by phase delay vs. port 1
    try:
        TxN = port_order[1]
        RxN = port_order[3]

        if Raw_LowFreq_Matrix[TxN - 1, 0] > Raw_LowFreq_Matrix[0, TxN - 1]:
            vector1 = sch[:, TxN - 1, 0]
        else:
            vector1 = sch[:, 0, TxN - 1]
        if Raw_LowFreq_Matrix[RxN - 1, 0] > Raw_LowFreq_Matrix[0, RxN - 1]:
            vector2 = sch[:, RxN - 1, 0]
        else:
            vector2 = sch[:, 0, RxN - 1]

        Floc = F.copy()
        if Floc[0] == 0:
            vector1 = vector1[1:]
            vector2 = vector2[1:]
            Floc = Floc[1:]

        phase_delay1 = -1.0 * np.unwrap(np.angle(vector1)) / (Floc * 2 * np.pi)
        phase_delay2 = -1.0 * np.unwrap(np.angle(vector2)) / (Floc * 2 * np.pi)

        # scalar phase delay = mean over the 0.25:0.75 band (1-based inclusive slice)
        quarter_size = _mround(len(Floc) / 4.0)
        three_quarter_size = _mround(len(Floc) * 3.0 / 4.0)
        mean_phase_delay1 = float(np.mean(phase_delay1[quarter_size - 1:three_quarter_size]))
        mean_phase_delay2 = float(np.mean(phase_delay2[quarter_size - 1:three_quarter_size]))

        # confidence check: far-side delay should be ~2x near-side to decide
        if max(mean_phase_delay1, mean_phase_delay2) > min(mean_phase_delay1, mean_phase_delay2) * 2:
            far_side = int(np.argmax([mean_phase_delay1, mean_phase_delay2])) + 1
            if far_side == 1:
                port_order[1], port_order[3] = port_order[3], port_order[1]
        else:
            print('Did not use phase delay in auto-port discovery since the phase delay '
                  'of Near End and Far End are similar')
    except Exception as ME_msg:
        print(str(ME_msg))
        print('Unable to use phase delay to determine port order')

    # Flip which side is the Rx side
    if flip_victim:
        port_order = [port_order[2], port_order[3], port_order[0], port_order[1]]

    print(f'Auto Port Order: [{" ".join(str(p) for p in port_order)}]')
    return port_order


if __name__ == '__main__':
    # smoke test: ports 1<->2 and 3<->4 connected, no crosstalk -> [1, 3, 2, 4]
    nf = 50
    F = np.linspace(0, 40e9, nf)
    sch = np.zeros((nf, 4, 4), dtype=complex)
    for d in range(4):
        sch[:, d, d] = 0.05
    sch[:, 0, 1] = sch[:, 1, 0] = 0.8
    sch[:, 2, 3] = sch[:, 3, 2] = 0.8
    print('port_order =', auto_port_order(sch, F))
