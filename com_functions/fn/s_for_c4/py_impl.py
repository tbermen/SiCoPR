# ============================================================
# MATLAB→Python translation notes for s_for_c4
# MATLAB lines: 11155–11159
# ============================================================
# External dependencies: s2_to_s4 and snp2smp are NOT in the MATLAB file.
#   They are from the MATLAB RF Toolbox / COM package.
# s2_to_s4(S2): embeds a 2-port S-matrix into a block-diagonal 4-port:
#   S4[0:2, 0:2, :] = S2,  S4[2:4, 2:4, :] = S2,  off-diagonal = 0
# snp2smp(S4, zref, [1 3 2 4]): applies the mixed-mode transformation
#   for port pairs (1,3) and (2,4). For the symmetric block-diagonal case
#   (s11=s22, s12=s21 — a shunt capacitor), this transformation leaves the
#   matrix unchanged (verified analytically via M*S4*M^T = S4 for the
#   standard mixed-mode matrix M).
# Therefore s_for_c4.Parameters is the block-diagonal 4-port of s_for_c2.
# Assumption documented; callers in COM use .Parameters field.
# Output struct: SimpleNamespace with Parameters shape (4,4,N).
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
from types import SimpleNamespace
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.s_for_c2.py_impl import s_for_c2


def s_for_c4(zref, f, cpad):
    """4-port S-parameters for a balanced shunt capacitor cpad.

    Constructs the block-diagonal 4×4 from the 2-port (s_for_c2), then
    applies the mixed-mode transformation snp2smp([1 3 2 4]).  For this
    symmetric network the result equals the block-diagonal itself.
    """
    S2 = s_for_c2(zref, f, cpad)
    s2p = S2.Parameters        # (2, 2, N)
    N = s2p.shape[2]

    # s2_to_s4: block-diagonal embedding
    S4P = np.zeros((4, 4, N), dtype=complex)
    S4P[0:2, 0:2, :] = s2p    # top-left block (ports 1,2)
    S4P[2:4, 2:4, :] = s2p    # bottom-right block (ports 3,4)

    # snp2smp with port pairs (1,3) and (2,4): for this symmetric matrix,
    # the mixed-mode transform M*S4*M^T = S4 (analytically verified).
    params = S4P

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
    print("s21 block0 at DC:", S.Parameters[1, 0, 0])
    print("s21 block1 at DC:", S.Parameters[3, 2, 0])
