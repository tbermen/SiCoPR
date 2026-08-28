# ============================================================
# MATLAB→Python translation notes for s_for_c2
# MATLAB lines: 11147–11153
# ============================================================
# 1-based vs 0-based: S_Parameters(1,1,:) → params[0,0,:], etc.
# sparameters(): MATLAB RF Toolbox object → SimpleNamespace with
#   .Parameters (2,2,N complex), .Frequencies (N,), .Impedance scalar.
# Formula: shunt capacitor cpad between signal and ground.
#   Y = jωC = j*2π*f*cpad
#   s11 = s22 = -Y*zref / (2 + Y*zref) = -jωC*zref / (2+jωC*zref)
#   s21 = s12 = 2 / (2 + Y*zref)
# Limits: f→0: s11→0, s21→1 (capacitor open at DC) ✓
#         f→∞: s11→-1, s21→0 (capacitor shorted at high freq) ✓
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
from types import SimpleNamespace
import numpy as np


def s_for_c2(zref, f, cpad):
    """2-port S-parameters for a shunt capacitor cpad.

    Returns SimpleNamespace with:
      .Parameters  — (2,2,N) complex array
      .Frequencies — frequency array (N,)
      .Impedance   — reference impedance scalar
    """
    f = np.asarray(f, dtype=float).ravel()
    N = len(f)

    jw_C_Z = 1j * 2.0 * np.pi * f * cpad * zref   # jωC*zref, shape (N,)
    denom = 2.0 + jw_C_Z

    params = np.empty((2, 2, N), dtype=complex)
    params[0, 0, :] = -jw_C_Z / denom    # MATLAB line 11149
    params[1, 1, :] = -jw_C_Z / denom    # MATLAB line 11150
    params[1, 0, :] = 2.0 / denom        # MATLAB line 11151
    params[0, 1, :] = 2.0 / denom        # MATLAB line 11152

    S = SimpleNamespace()
    S.Parameters = params
    S.Frequencies = f
    S.Impedance = zref
    return S


if __name__ == "__main__":
    import numpy as np
    f = np.array([0.0, 1e9, 10e9])
    S = s_for_c2(50.0, f, 1e-12)
    print("s21 at DC:", S.Parameters[1, 0, 0])      # expected ~1
    print("s21 at 10GHz:", S.Parameters[1, 0, 2])   # attenuated
