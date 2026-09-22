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



def _lfsr(s, t):
    """Linear feedback shift register (MATLAB lines 4195-4211).

    s: initial seed (list/array of bits)
    t: tap positions (1-based, as in MATLAB)
    Returns (seq, c) where seq is the output bit sequence and c is the state history.
    """
    s = list(int(b) for b in s)
    n = len(s)
    t = [int(x) - 1 for x in t]  # convert 1-based taps to 0-based
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
    seq = c_arr[:, n - 1]  # MATLAB: c(:, n) → 0-based: c[:,n-1]
    return seq, c_arr


def _pam(data):
    """Grey-coded PAM4 mapping (MATLAB lines 4213-4225).

    Maps NRZ pairs to PAM4 levels: (-1,-1)→-1, (-1,1)→-1/3, (1,1)→1/3, (1,-1)→1.
    """
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


def PRBS13Q():
    """Generate PRBS13Q PAM4 sequence (MATLAB lines 4174-4192).

    Returns (seq, syms, syms_nrz) where:
      seq      : PAM4 symbols in {-1, -1/3, 1/3, 1}
      syms     : integer symbols in {0, 1, 2, 3}
      syms_nrz : NRZ bit sequence (same as seq_nrz = 2*(bits-0.5))
    """
    taps = [13, 12, 2, 1]
    seed = [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    seq_bits, _ = _lfsr(seed, taps)
    seq_nrz = 2.0 * (seq_bits - 0.5)
    seq = _pam(seq_nrz)

    syms = np.zeros(len(seq), dtype=int)
    syms[_mround_arr(2 * (seq + 1)) / 2 == 2] = 3
    syms[_mround_arr(2 * (seq + 1)) / 2 == 1.5] = 2
    syms[_mround_arr(2 * (seq + 1)) / 2 == 0.5] = 1
    syms[_mround_arr(2 * (seq + 1)) / 2 == 0] = 0

    syms_nrz = seq_nrz
    return seq, syms, syms_nrz
