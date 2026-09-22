import numpy as np

def _mround_arr(x):
    """MATLAB round() on an array: halves go away from zero, where np.round
    takes them to even.

    Only exact ties are corrected. Adding 0.5 and truncating would be wrong:
    0.49999999999999994 + 0.5 is exactly 1.0 in double precision, so that form
    rounds the largest double below a half up to 1 where MATLAB gives 0.
    """
    x = np.asarray(x, dtype=float)
    tie = np.abs(x - np.trunc(x)) == 0.5
    return np.where(tie, np.trunc(x) + np.copysign(1.0, x), np.round(x))

from scipy.signal import lfilter
from types import SimpleNamespace


def _lfsr(s, t):
    s = [int(b) for b in s]
    n = len(s)
    t = [int(x) - 1 for x in t]  # 1-based → 0-based
    m = len(t)
    c = [s[:]]
    for _ in range(2 ** n - 2):
        b = [0] * m
        b[0] = s[t[0]] ^ s[t[1]]
        for i in range(m - 2):
            b[i + 1] = s[t[i + 2]] ^ b[i]
        for j in range(n - 1):
            s[n - 1 - j] = s[n - 2 - j]
        s[0] = b[m - 2]
        c.append(s[:])
    c_arr = np.array(c, dtype=int)
    return c_arr[:, n - 1]  # output bit column


def _pam(data):
    data = np.asarray(data, dtype=float)
    n_pairs = int(np.floor(len(data) / 2))
    dataout = np.zeros(n_pairs)
    for i in range(n_pairs):
        pair = data[2 * i: 2 * i + 2]
        if np.array_equal(pair, [-1, -1]):
            dataout[i] = -1.0
        elif np.array_equal(pair, [-1, 1]):
            dataout[i] = -1.0 / 3.0
        elif np.array_equal(pair, [1, 1]):
            dataout[i] = 1.0 / 3.0
        elif np.array_equal(pair, [1, -1]):
            dataout[i] = 1.0
    return dataout


def _PRBS13Q():
    seq_bits = _lfsr([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1], [13, 12, 2, 1])
    seq_nrz = 2.0 * (seq_bits - 0.5)
    seq = _pam(seq_nrz)
    syms = np.zeros(len(seq), dtype=int)
    syms[_mround_arr(2 * (seq + 1)) / 2 == 2] = 3
    syms[_mround_arr(2 * (seq + 1)) / 2 == 1.5] = 2
    syms[_mround_arr(2 * (seq + 1)) / 2 == 0.5] = 1
    return seq, syms, seq_nrz


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

    P_3 = float(np.mean(Bit_stream_response[icent3 - M:icent3 + M + 1]))
    P_0 = float(np.mean(Bit_stream_response[icent0 - M:icent0 + M + 1]))
    VMA = P_3 - P_0

    return SimpleNamespace(P_3=P_3, P_0=P_0, VMA=VMA)
