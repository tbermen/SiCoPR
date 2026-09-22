# ============================================================
# MATLAB→Python translation notes for capture_RIL_RILN
# MATLAB lines: 5087–5245
# ============================================================
# Computes Reflectionless Insertion Loss (RIL) and
# Reflective Insertion Loss Noise (RILN) from differential S-params.
#
# Input: chdata list (uses chdata[0]: sdd11_orig, sdd22_orig, sdd12_orig, sdd21_orig, faxis)
# Impedance: 100 Ω (hardcoded per MATLAB)
# idx_start: 1 if faxis[0]==0 else 0 (skip DC)
#
# Quadratic for rho_port2:
#   a*rho^2 + b*rho + c = 0 → solutions via quadratic formula.
#   Exactly one solution must give Re(Z) > 0; MATLAB errors otherwise (L5211).
# rho_port1 = conj(Sdd11 + rho_port2*Sdd21*Sdd12/(1 - rho_port2*Sdd22))
# RIL formula: see equations in function body.
# ============================================================

import numpy as np
from types import SimpleNamespace


def capture_RIL_RILN(chdata):
    """RIL/RILN computation (MATLAB lines 5087-5245).

    Uses chdata[0] differential S-parameters to compute Reflectionless
    Insertion Loss (RIL) and Reflective Insertion Loss Noise (RILN).

    Returns SimpleNamespace with fields:
      RIL, RIL_dB, RILN, RILN_dB, Z_port1, Z_port2,
      rho_port1, rho_port2, freq.
    """
    if isinstance(chdata, list):
        ch = chdata[0]
    else:
        ch = chdata

    faxis = np.asarray(ch.faxis, dtype=float)
    Z0 = 100.0  # hardcoded per MATLAB

    Sdd11 = np.asarray(ch.sdd11_orig, dtype=complex).ravel()
    Sdd12 = np.asarray(ch.sdd12_orig, dtype=complex).ravel()
    Sdd21 = np.asarray(ch.sdd21_orig, dtype=complex).ravel()
    Sdd22 = np.asarray(ch.sdd22_orig, dtype=complex).ravel()

    if len(Sdd11) != 2 and len(faxis) > 0 and faxis[0] != 0:
        # Validate 2-port
        pass

    # Skip DC if first frequency is 0
    idx_start = 1 if faxis[0] == 0 else 0

    S11 = Sdd11[idx_start:]
    S12 = Sdd12[idx_start:]
    S21 = Sdd21[idx_start:]
    S22 = Sdd22[idx_start:]

    # Quadratic coefficients for rho_port2
    a = -S22 + S11 * S22 * np.conj(S11) - S21 * S12 * np.conj(S11)
    b = (1 + S22 * np.conj(S22)
         + S11 * S22 * np.conj(S21) * np.conj(S12)
         - S21 * S12 * np.conj(S21) * np.conj(S12)
         - S11 * S22 * np.conj(S11) * np.conj(S22)
         + S12 * S21 * np.conj(S11) * np.conj(S22)
         - S11 * np.conj(S11))
    c = (-np.conj(S22)
         - S11 * np.conj(S21) * np.conj(S12)
         + S11 * np.conj(S11) * np.conj(S22))

    disc = np.sqrt(b ** 2 - 4 * a * c + 0j)
    solution_1 = (-b + disc) / (2 * a)
    solution_2 = (-b - disc) / (2 * a)

    # Select solution with Re(Z) > 0
    Z_sol1 = (solution_1 * np.conj(Z0) + Z0) / (1 - solution_1)
    Z_sol2 = (solution_2 * np.conj(Z0) + Z0) / (1 - solution_2)

    rho_port2 = np.zeros(len(solution_1), dtype=complex)
    for k in range(len(solution_1)):
        re1 = float(np.real(Z_sol1[k]))
        re2 = float(np.real(Z_sol2[k]))
        if re1 > 0 and re2 <= 0:
            rho_port2[k] = solution_1[k]
        elif re2 > 0 and re1 <= 0:
            rho_port2[k] = solution_2[k]
        else:
            # MATLAB raises here (L5211); it does NOT choose between them.
            # Picking the smaller |rho| continued past a point the reference
            # refuses, and every number downstream of it was unverifiable --
            # in the S11=S22=0 case it was literally NaN.
            # COM Octave: S11=S22=0, S21=S12 lossy -> "error: An odd case has
            # occured. Please contact the tool developer."; likewise for
            # S11=S22=0.5 with S21=S12=sqrt(0.75).
            raise ValueError('An odd case has occured. '
                             'Please contact the tool developer.')

    rho_port1 = np.conj(S11 + (rho_port2 * S21 * S12) / (1 - rho_port2 * S22))

    Z_port1 = (rho_port1 * np.conj(Z0) + Z0) / (1 - rho_port1)
    Z_port2 = (rho_port2 * np.conj(Z0) + Z0) / (1 - rho_port2)

    # RIL formula
    abs1 = np.abs(1 - rho_port1 * np.conj(rho_port1))
    abs2 = np.abs(1 - rho_port2 * np.conj(rho_port2))
    numer = (np.conj((1 - np.conj(rho_port1)) * (np.sqrt(abs1) / np.abs(1 - rho_port1)))
             * (S21 * (1 - rho_port2 * S22) + (S22 - np.conj(rho_port2)) * rho_port2 * S21))
    denom = (((1 - np.conj(rho_port2)) * (np.sqrt(abs2) / np.abs(1 - rho_port2)))
             * (-rho_port1 * rho_port2 * S21 * S12 + (1 - rho_port2 * S22) * (1 - rho_port1 * S11)))

    RIL = numer / denom
    RILN = RIL - S21
    # No epsilon floor: MATLAB is 20*log10(abs(Sdd21)) - 20*log10(abs(RIL)).
    # Adding eps moved every value, not just the zeros -- d(20*log10 x)/dx is
    # 20/(x*ln10), so at |RIL| ~ 0.8 an eps shifts the dB by ~2.4e-15, and the
    # measured gap on an ordinary lossy 2-port was 4.44e-15 dB. At a true zero
    # it replaced the reference's answer outright.
    # COM Octave, S21=S12=0: RIL_dB = -Inf (the eps form gave -313.07) and
    # RILN_dB = NaN (-Inf minus -Inf; the eps form gave 0).
    RILN_dB = 20 * np.log10(np.abs(S21)) - 20 * np.log10(np.abs(RIL))

    rs = SimpleNamespace()
    rs.RIL = RIL
    rs.RIL_dB = 20 * np.log10(np.abs(RIL))
    rs.RILN = RILN
    rs.RILN_dB = RILN_dB
    rs.Z_port1 = Z_port1
    rs.Z_port2 = Z_port2
    rs.rho_port1 = rho_port1
    rs.rho_port2 = rho_port2
    rs.freq = faxis[idx_start:]
    return rs
