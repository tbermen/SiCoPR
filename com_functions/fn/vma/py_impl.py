import numpy as np
from com_functions.fn.PRBS13Q.py_impl import PRBS13Q as _PRBS13Q
from com_functions.fn.pam.py_impl import pam as _pam

from scipy.signal import lfilter
from types import SimpleNamespace


def _strfind_int(arr, pattern):
    """Find all 0-based positions where pattern occurs in arr."""
    arr = np.asarray(arr)
    n, m = len(arr), len(pattern)
    return np.array([i for i in range(n - m + 1) if np.array_equal(arr[i:i + m], pattern)],
                    dtype=int)


def vma(PR, M):
    """Compute Vertical Margin with Alignment (VMA) for PAM4 (MATLAB lines 11337-11365).

    PR: pulse response (1-D array), M: samples per UI.
    Returns SimpleNamespace with P_3, P_0, VMA fields.
    """
    PR = np.asarray(PR, dtype=float).ravel()
    M = int(M)

    seq, syms, _ = _PRBS13Q()
    symbols = seq

    imaxPR = int(np.argmax(PR))  # 0-based

    # Positions of 7 consecutive 3s in syms (0-based)
    pos_3x7 = _strfind_int(syms, [3, 3, 3, 3, 3, 3, 3])
    # MATLAB: indx_S3x7_start = M*(pos-1) + imaxPR (1-based) → Python 0-based: M*pos + imaxPR
    indx_S3x7_start = M * pos_3x7 + imaxPR
    indx_S3x7_end = M * (pos_3x7 + 6) + imaxPR  # MATLAB: M*(pos+5)+imaxPR (1-based)

    # Positions of 6 consecutive 0s in syms (0-based)
    pos_0x6 = _strfind_int(syms, [0, 0, 0, 0, 0, 0])
    # MATLAB: M*pos - 1 + imaxPR (1-based) → Python 0-based: M*(pos_0+1) + imaxPR_0 - 1
    indx_S0x6_start = M * (pos_0x6 + 1) + imaxPR - 1
    indx_S0x6_end = M * (pos_0x6 + 5) + imaxPR  # MATLAB: M*(pos+4)+imaxPR (1-based)

    # Upsample symbols by M: kron(symbols, [1, 0, 0, ..., 0])
    unit_pulse = np.zeros(M)
    unit_pulse[0] = 1
    shifting_vector = np.kron(symbols, unit_pulse)

    # Superposition response: filter(PR, 1, shifting_vector) — PR is the FIR kernel
    Bit_stream_response = lfilter(PR, [1.0], shifting_vector)

    # Center indices for 3s and 0s patterns (take first match)
    icent3 = int(np.floor((indx_S3x7_end - indx_S3x7_start) / 2 + indx_S3x7_start)[0])
    icent0 = int(np.floor((indx_S0x6_end - indx_S0x6_start) / 2 + indx_S0x6_start)[0])

    # MATLAB indexes Bit_stream_response((icent-M):(icent+M)) directly, so a
    # window that runs off either end is an error there, not a shorter mean.
    # COM Octave, M=1 with a 9000-sample PR peaking at sample 8901:
    #   "error: Bit_stream_response(11732): out of bound 4095".
    # A numpy slice clips instead, and an empty slice made P_3 and P_0 NaN
    # with no sign that the measurement had left the bit stream.
    n_bsr = len(Bit_stream_response)
    for icent in (icent3, icent0):
        if icent - M < 0 or icent + M >= n_bsr:
            raise IndexError(
                'vma: measurement window Bit_stream_response(%d..%d) lies '
                'outside the %d-sample bit stream'
                % (icent - M + 1, icent + M + 1, n_bsr))

    P_3 = float(np.mean(Bit_stream_response[icent3 - M:icent3 + M + 1]))
    P_0 = float(np.mean(Bit_stream_response[icent0 - M:icent0 + M + 1]))
    VMA = P_3 - P_0

    return SimpleNamespace(P_3=P_3, P_0=P_0, VMA=VMA)
