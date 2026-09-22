# ============================================================
# MATLAB→Python translation notes for read_p2_s2params
# MATLAB lines: 10259–10390
# ============================================================
# Reads 2-port Touchstone, converts single-ended → differential mode.
# T = [[1,1],[1,-1]]; W = T * (S / T) = T * S * inv(T).
# D[i] = W for each frequency point i (0-based in Python).
# SDD(:,1,1) = D(:,2,2) → Python D[:,1,1]
# SDC(:,1,1) = D(:,2,1) → Python D[:,1,0]
# SCC(:,1,1) = D(:,1,1) → Python D[:,0,0]
# SCD(:,1,1) = D(:,1,2) → Python D[:,0,1]
# rangelimit and read_Nport_touchstone are called, not re-inlined: the private
# copies that used to sit here had drifted from the shared functions.
# Plotting skipped (DISPLAY_WINDOW handled silently).
# ============================================================

import numpy as np
from com_functions.fn.rangelimit.py_impl import rangelimit as _rangelimit
from com_functions.fn.read_Nport_touchstone.py_impl import read_Nport_touchstone as _read_Nport_touchstone
from types import SimpleNamespace


def read_p2_s2params(infile, plot_ini_s_params, plot_dif_s_params, ports, OP, param):
    """Read 2-port Touchstone and convert to differential mode (MATLAB lines 10259-10390).

    Returns (data, SDD, SDC, SCC, SCD).
    All SXX arrays have shape (nfreq, 1, 1).
    """
    if np.size(ports) == 0:          # `not ports` raises on an ndarray
        ports = [1, 2]
    ports = [1, 2]  # MATLAB overrides ports to [1,2]

    # MATLAB asks for two outputs here; the reader's third is the resolved
    # port order, which this caller does not use.
    sch, freq, _ = _read_Nport_touchstone(infile, ports, float(param.Z0))
    # rangelimit returns its OWN param. MATLAB passes param by value, so the
    # caller's struct is untouched; the private copy this used to call wrote
    # param.flim straight back into the caller's object. COM Octave, a file
    # ending at 40 GHz read with param.flim = 67e9: data.flim comes back
    # 40e9 while the caller's param.flim is still 67e9.
    sch, freq, limited, param_out = _rangelimit(sch, freq, param, OP)

    nfreq = len(freq)
    T = np.array([[1.0, 1.0], [1.0, -1.0]])

    D = np.zeros((nfreq, 2, 2), dtype=complex)
    for i in range(nfreq):
        S = sch[i, :, :]
        # MATLAB's S/T is mrdivide, which SOLVES rather than multiplying by an
        # inverse: S/T == (T.'\S.').'. T @ S @ inv(T) is the same matrix in
        # exact arithmetic and not in floating point.
        D[i] = T @ np.linalg.solve(T.T, S.T).T

    # Extract mixed-mode S-params (MATLAB 1-based → Python 0-based)
    SDD = np.zeros((nfreq, 1, 1), dtype=complex)
    SDC = np.zeros((nfreq, 1, 1), dtype=complex)
    SCC = np.zeros((nfreq, 1, 1), dtype=complex)
    SCD = np.zeros((nfreq, 1, 1), dtype=complex)

    SDD[:, 0, 0] = D[:, 1, 1]  # MATLAB D(:,2,2)
    SDC[:, 0, 0] = D[:, 1, 0]  # MATLAB D(:,2,1)
    SCC[:, 0, 0] = D[:, 0, 0]  # MATLAB D(:,1,1)
    SCD[:, 0, 0] = D[:, 0, 1]  # MATLAB D(:,1,2)

    data = SimpleNamespace()
    data.m = sch
    data.freq = freq
    data.flim = param_out.flim
    data.limited = limited

    return data, SDD, SDC, SCC, SCD
