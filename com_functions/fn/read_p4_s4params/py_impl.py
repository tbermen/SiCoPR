# ============================================================
# MATLAB→Python translation notes for read_p4_s4params
# MATLAB lines: 10391–10524
# ============================================================
# Reads 4-port Touchstone, applies TX/RX skew correction, converts to mixed-mode.
# T = [[1,1,0,0],[1,-1,0,0],[0,0,1,1],[0,0,1,-1]]
# sigma_matrix for each frequency point based on TX/RX p/n skew.
# Snew = sigma_matrix .* S (elementwise, not matrix multiply)
# W = T * (Snew / T)  -- mrdivide, a solve, not a multiply by inv(T)
# D matrix indexing (MATLAB 1-based):
#   SDD(1,1) = D(2,2); SDD(2,2) = D(4,4); SDD(1,2) = D(2,4); SDD(2,1) = D(4,2)
#   SDC(1,1) = D(2,1); SDC(2,2) = D(4,3); SDC(1,2) = D(2,3); SDC(2,1) = D(4,1)
#   SCC(1,1) = D(1,1); SCC(2,2) = D(3,3); SCC(1,2) = D(1,3); SCC(2,1) = D(3,1)
#   SCD(1,1) = D(1,2); SCD(2,2) = D(3,4); SCD(1,2) = D(1,4); SCD(2,1) = D(3,2)
# ============================================================

# rangelimit and read_Nport_touchstone are called, not re-inlined: the private
# copies that used to sit here had drifted from the shared functions, exactly
# as read_p2_s2params's had. The reader copy defaulted nport to 4 rather than 2
# for an extension with no digit and did not reject a file it parsed no data
# from; the rangelimit copy wrote param.flim straight back into the CALLER's
# object, where MATLAB passes param by value.
# COM Octave, a 50 GHz file read with param.flim = 100e9: data.flim comes back
# 50e9 while the caller's param.flim is still 100e9 -- the port left the
# caller holding 50e9.

import numpy as np
from com_functions.fn.rangelimit.py_impl import rangelimit as _rangelimit
from com_functions.fn.read_Nport_touchstone.py_impl import read_Nport_touchstone as _read_Nport_touchstone
from types import SimpleNamespace


def read_p4_s4params(infile, plot_ini_s_params, plot_dif_s_params, ports, OP, param):
    """Read 4-port Touchstone and convert to mixed-mode (MATLAB lines 10391-10524).

    Returns (data, SDD, SDC, SCC, SCD, ports).
    All SXX arrays have shape (nfreq, 2, 2).
    r4p15p0: no [1 3 2 4] default; empty ports triggers auto-detection and the
    resolved order is returned as the 6th output.
    """
    sch, freq, ports = _read_Nport_touchstone(infile, ports, float(param.Z0))
    # rangelimit returns its OWN param; MATLAB passes param by value, so the
    # caller's struct is untouched.
    sch, freq, limited, param_out = _rangelimit(sch, freq, param, OP)

    nfreq = len(freq)

    # Skew parameters (default 0 if not set)
    Txpskew = float(getattr(param, 'Txpskew', 0.0))
    Txnskew = float(getattr(param, 'Txnskew', 0.0))
    Rxpskew = float(getattr(param, 'Rxpskew', 0.0))
    Rxnskew = float(getattr(param, 'Rxnskew', 0.0))

    T = np.array([[1.0, 1.0, 0.0, 0.0],
                  [1.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0, -1.0]])

    D = np.zeros((nfreq, 4, 4), dtype=complex)
    for i in range(nfreq):
        f = freq[i]
        # MATLAB's Sigfct is `@(sigma2,sigma1,sigma4,sigma3)...`: the parameter
        # NAMES are transposed on purpose ("need to swap sigma for 1 and 3 and
        # 2 and 4", RIM 12/29/2023), so calling it with (Txp, Txn, Rxp, Rxn)
        # binds sigma1=Txn, sigma2=Txp, sigma3=Rxn, sigma4=Rxp. The port read
        # the names in call order and built the matrix with 1<->2 and 3<->4
        # swapped, which is invisible while the p and n skews match and wrong
        # as soon as they do not.
        # COM Octave, Txpskew=3 ps, Txnskew=-1 ps, Rx skews 0: max|dSDC| and
        # max|dSCD| reach 0.1255 (SDC and SCD are O(0.1) here), max|dSDD|
        # 5.67e-4.
        s1 = np.exp(2j * np.pi * f * Txnskew * 1e-12)   # MATLAB sigma1
        s2 = np.exp(2j * np.pi * f * Txpskew * 1e-12)   # MATLAB sigma2
        s3 = np.exp(2j * np.pi * f * Rxnskew * 1e-12)   # MATLAB sigma3
        s4 = np.exp(2j * np.pi * f * Rxpskew * 1e-12)   # MATLAB sigma4
        sigma_matrix = np.array([
            [s1 ** 2,   s1 * s2, s1 * s3, s1 * s4],
            [s1 * s2,   s2 ** 2, s2 * s3, s2 * s4],
            [s1 * s3,   s2 * s3, s3 ** 2, s3 * s4],
            [s1 * s4,   s2 * s4, s3 * s4, s4 ** 2],
        ])
        S = sch[i, :, :]
        Snew = sigma_matrix * S  # elementwise (Sigfct .* S)
        # MATLAB's Snew/T is mrdivide, which SOLVES rather than multiplying by
        # an inverse: Snew/T == (T.'\Snew.').'. T @ Snew @ inv(T) is the same
        # matrix in exact arithmetic and not in floating point.
        D[i] = T @ np.linalg.solve(T.T, Snew.T).T

    # Extract mixed-mode S-params (0-based Python, MATLAB 1-based → subtract 1)
    SDD = np.zeros((nfreq, 2, 2), dtype=complex)
    SDD[:, 0, 0] = D[:, 1, 1]   # D(2,2)
    SDD[:, 1, 1] = D[:, 3, 3]   # D(4,4)
    SDD[:, 0, 1] = D[:, 1, 3]   # D(2,4)
    SDD[:, 1, 0] = D[:, 3, 1]   # D(4,2)

    SDC = np.zeros((nfreq, 2, 2), dtype=complex)
    SDC[:, 0, 0] = D[:, 1, 0]   # D(2,1)
    SDC[:, 1, 1] = D[:, 3, 2]   # D(4,3)
    SDC[:, 0, 1] = D[:, 1, 2]   # D(2,3)
    SDC[:, 1, 0] = D[:, 3, 0]   # D(4,1)

    SCC = np.zeros((nfreq, 2, 2), dtype=complex)
    SCC[:, 0, 0] = D[:, 0, 0]   # D(1,1)
    SCC[:, 1, 1] = D[:, 2, 2]   # D(3,3)
    SCC[:, 0, 1] = D[:, 0, 2]   # D(1,3)
    SCC[:, 1, 0] = D[:, 2, 0]   # D(3,1)

    SCD = np.zeros((nfreq, 2, 2), dtype=complex)
    SCD[:, 0, 0] = D[:, 0, 1]   # D(1,2)
    SCD[:, 1, 1] = D[:, 2, 3]   # D(3,4)
    SCD[:, 0, 1] = D[:, 0, 3]   # D(1,4)
    SCD[:, 1, 0] = D[:, 2, 1]   # D(3,2)

    data = SimpleNamespace()
    data.m = sch
    data.freq = freq
    data.flim = param_out.flim
    data.limited = limited

    return data, SDD, SDC, SCC, SCD, ports
