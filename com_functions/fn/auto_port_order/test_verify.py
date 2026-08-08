# ============================================================
# MATLAB GROUND TRUTH for auto_port_order (com_ieee8023_4p15p0.m L4932-5057)
# ------------------------------------------------------------
# The algorithm:
#   1. take |S| at the first frequency point, zero the diagonal,
#      symmetrize (reciprocity),
#   2. for each column find the strongest-coupled port (its "connection"),
#   3. build port_order = [1, min(others), conn(1), max(others)],
#   4. optionally swap entries 2 & 4 by a phase-delay near/far test,
#   5. optionally flip the victim side.
# Returned indices are 1-based (e.g. the classic [1 3 2 4]).
#
# Expected values below are derived by hand-executing the algorithm
# on each synthetic S-cube (no MATLAB run needed — pure topology):
#   * pairs (1<->2, 3<->4), no crosstalk  -> ConnectedPorts=[2,1,4,3] -> [1 3 2 4]
#   * pairs (1<->3, 2<->4), no crosstalk  -> ConnectedPorts=[3,4,1,2] -> [1 2 3 4]
#   * flip_victim on [1 3 2 4]            -> [2 4 1 3]
#   * S13 path given a large phase delay, S14 ~zero -> swap -> [1 4 2 3]
#   * all through energy < 0.1            -> ValueError (low energy)
# ============================================================

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.auto_port_order.py_impl import auto_port_order


def _make_sch(pairs, nf=64, fmax=40e9, crosstalk=None):
    """Build a synthetic 4-port S-cube.

    pairs: list of (a,b) 1-based connected port pairs with strong through.
    crosstalk: optional dict {(a,b): complex_array or scalar} added to sch[:,a-1,b-1].
    """
    F = np.linspace(0, fmax, nf)
    sch = np.zeros((nf, 4, 4), dtype=complex)
    for d in range(4):
        sch[:, d, d] = 0.05  # return loss (zeroed by the routine)
    for (a, b) in pairs:
        sch[:, a - 1, b - 1] = 0.8
        sch[:, b - 1, a - 1] = 0.8
    if crosstalk:
        for (a, b), v in crosstalk.items():
            sch[:, a - 1, b - 1] = sch[:, a - 1, b - 1] + v
    return sch, F


def test_nominal_1234_swap_order():
    """Pairs (1<->2, 3<->4) with no crosstalk -> classic [1, 3, 2, 4]."""
    sch, F = _make_sch([(1, 2), (3, 4)])
    assert auto_port_order(sch, F) == [1, 3, 2, 4]


def test_alternate_topology_identity():
    """Pairs (1<->3, 2<->4) -> [1, 2, 3, 4]."""
    sch, F = _make_sch([(1, 3), (2, 4)])
    assert auto_port_order(sch, F) == [1, 2, 3, 4]


def test_flip_victim():
    """flip_victim reorders [1 3 2 4] -> [2 4 1 3]."""
    sch, F = _make_sch([(1, 2), (3, 4)])
    assert auto_port_order(sch, F, flip_victim=1) == [2, 4, 1, 3]


def test_phase_delay_triggers_swap():
    """A large phase delay on the S13 path (near side) forces the 2&4 swap.

    Topology pairs (1<->2, 3<->4) -> base order [1 3 2 4] with TxN=3, RxN=4.
    Give S13/S31 a real group delay (so port 3's coupling to port 1 looks
    'far') and S14/S41 ~zero phase. far_side==1 -> swap -> [1 4 2 3].
    """
    nf = 64
    F = np.linspace(0, 40e9, nf)
    tau = 20e-12
    s13 = 0.01 * np.exp(-1j * 2 * np.pi * F * tau)
    s14 = 0.01 * np.ones(nf)
    sch, F = _make_sch([(1, 2), (3, 4)], nf=nf,
                       crosstalk={(1, 3): s13, (3, 1): s13,
                                  (1, 4): s14, (4, 1): s14})
    assert auto_port_order(sch, F) == [1, 4, 2, 3]


def test_low_energy_raises():
    """All through coupling below MinThruEnergy=0.1 -> ValueError."""
    F = np.linspace(0, 40e9, 32)
    sch = np.zeros((32, 4, 4), dtype=complex)
    for d in range(4):
        sch[:, d, d] = 0.05
    sch[:, 0, 1] = sch[:, 1, 0] = 0.02  # weak
    sch[:, 2, 3] = sch[:, 3, 2] = 0.02
    with pytest.raises(ValueError):
        auto_port_order(sch, F)


def test_non_4port_raises():
    """Routine only supports 4-port cubes."""
    F = np.linspace(0, 40e9, 16)
    sch = np.zeros((16, 2, 2), dtype=complex)
    with pytest.raises(ValueError):
        auto_port_order(sch, F)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
