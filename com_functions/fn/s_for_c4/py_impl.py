# ============================================================
# MATLAB→Python translation notes for s_for_c4
# MATLAB lines: 11155–11159
# ============================================================
# External dependencies: s2_to_s4 and snp2smp are NOT in the MATLAB file.
#   snp2smp is MATLAB RF Toolbox.  s2_to_s4 is defined nowhere — not in any
#   COM release 4p10p0..4p16p0, not in the 802-COM src tree, and not in RF
#   Toolbox — so the reference cannot execute this function and no oracle can
#   settle it.  Nothing in COM calls s_for_c4 either; it is dead code.
# s2_to_s4(S2): assumed to embed the 2-port into a block-diagonal 4-port,
#   S4[0:2,0:2,:] = S2, S4[2:4,2:4,:] = S2 — two uncoupled copies of the
#   shunt cap, one on raw ports 1-2 and one on raw ports 3-4.
# snp2smp(S, zref, [1 3 2 4]) with M == N == 4 terminates no port, so it is
#   the pure port permutation new(i,j) = old(p(i), p(j)), p = [1 3 2 4] —
#   the same operation the reference spells out at line 10050 as
#   `sch=sch(:,port_order,port_order)`, with the same default vector
#   (line 10434: `'Port Order', true, [1 3 2 4]  % [ tx+ tx- rx+ rx-]`).
#   It is NOT a mixed-mode transform and it is NOT a no-op: it swaps ports
#   2 and 3, turning the block diagonal into the COM [tx+ tx- rx+ rx-] form
#   where 1,2 are the Tx pair and 3,4 the Rx pair.  Omitting it returned a
#   4-port whose differential insertion loss Sdd21 was identically zero.
# Output struct: SimpleNamespace with Parameters shape (4,4,N).
# ============================================================
from types import SimpleNamespace
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.s_for_c2.py_impl import s_for_c2


def s_for_c4(zref, f, cpad):
    """4-port S-parameters for a balanced shunt capacitor cpad.

    Constructs the block-diagonal 4×4 from the 2-port (s_for_c2), then
    reorders the ports as snp2smp(...,[1 3 2 4]) does, into COM's
    [tx+ tx- rx+ rx-] convention.
    """
    S2 = s_for_c2(zref, f, cpad)
    s2p = S2.Parameters        # (2, 2, N)
    N = s2p.shape[2]

    # s2_to_s4: block-diagonal embedding
    S4P = np.zeros((4, 4, N), dtype=complex)
    S4P[0:2, 0:2, :] = s2p    # top-left block (raw ports 1,2)
    S4P[2:4, 2:4, :] = s2p    # bottom-right block (raw ports 3,4)

    # snp2smp(S.Parameters, zref, [1 3 2 4]) — MATLAB line 11650.  M == N,
    # so no port is terminated and this is `sch(port_order, port_order, :)`.
    perm = [0, 2, 1, 3]                       # 1-based [1 3 2 4]
    params = S4P[perm, :, :][:, perm, :]

    S = SimpleNamespace()
    S.Parameters = params
    S.Frequencies = np.asarray(f, dtype=float).ravel()
    S.Impedance = zref
    return S


if __name__ == "__main__":
    import numpy as np
    f = np.array([0.0, 1e9])
    S = s_for_c4(50.0, f, 1e-12)
    print("shape:", S.Parameters.shape)
    print("tx+ -> rx+ at DC:", S.Parameters[2, 0, 0])
    print("tx- -> rx- at DC:", S.Parameters[3, 1, 0])
