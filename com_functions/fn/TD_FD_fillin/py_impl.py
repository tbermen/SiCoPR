# ============================================================
# MATLAB→Python translation notes for TD_FD_fillin
# MATLAB lines: 4610–4674
# ============================================================
# Fills frequency-domain fields in chdata from the time-domain pulse response.
# MATLAB f=0:1/max(T):1/dt → np.arange(0, f_nyq+df/2, df) to include Nyquist.
# prr = sin(f*UI*pi)/(f*UI*pi) = np.sinc(f*UI) (numpy sinc is normalised).
# fd = fft(V), keep first half (positive frequencies).
# IL_conv uses only indices up to f75 (0.75*fb).
# SDDch first dim is frequency, indices [0,1] → [1,2] MATLAB port mapping.
# timeseries MATLAB object (STEP) not used outside this function — omitted.
# ============================================================

import math
import numpy as np
from scipy.signal import lfilter


def _factorial(k):
    """MATLAB factorial(): a double, so it overflows to Inf above 170!."""
    return math.inf if k > 170 else math.factorial(k)


def _bessel(n):
    # `for ii = 0:n` never runs for n < 0, so MATLAB never assigns `a` and the
    # function errors.  COM Octave: bessel(-1) -> "value on right hand side of
    # assignment is undefined".  Returning an empty array answered a call the
    # reference refuses.  Non-integer n is rejected by MATLAB factorial().
    if n < 0:
        raise ValueError('bessel: output is undefined for n < 0 (got %r)' % (n,))
    if n != int(n):
        raise ValueError('bessel: n must be a non-negative integer (got %r)' % (n,))
    n = int(n)
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        # COM Octave, bessel(90): a(1:10) are Inf.  Python's exact
        # math.factorial made them finite (~1.09e164) instead.
        a[ii] = (_factorial(2 * n - ii)
                 / (2 ** (n - ii) * _factorial(ii) * _factorial(n - ii)))
    return a


def _Bessel_Thomson_Filter(param, f, use_BT):
    f = np.asarray(f, dtype=float)
    # MATLAB `if use_BT` is true only for a non-empty value whose elements are
    # ALL non-zero, and length() is the LONGEST dimension, not the first.
    # COM Octave: use_BT=[] or [1 0] -> ones branch; f 2x3 -> ones(1,3);
    # f scalar -> 1 (len(f) raised TypeError).
    use = np.asarray(use_BT)
    if not (use.size and np.all(use)):
        return np.ones(_length(f))
    a = _bessel(param.BTorder)
    acoef = a[::-1]
    s = 1j * f / (param.fb_BT_cutoff * param.fb)
    return a[0] / np.polyval(acoef, s)


_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]


def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def _Butterworth_Filter(param, f, use_BW):
    f = np.asarray(f, dtype=float)
    # MATLAB `if use_BW` is true only for a non-empty value whose elements are
    # ALL non-zero.  COM Octave: use_BW=[] -> ones branch, [1 0] -> ones branch,
    # [1 1] -> filter branch.
    use = np.asarray(use_BW)
    if not (use.size and np.all(use)):
        # ones(1,length(f)): length() is the LONGEST dimension, not the first.
        # COM Octave: f 2x3 -> ones(1,3), three elements not six; f scalar -> 1
        # (len(f) raised TypeError on a scalar).
        return np.ones(_length(f))
    s = 1j * f / (param.fb_BW_cutoff * param.fb)
    return 1.0 / np.polyval(_BW_POLY, s)


def TD_FD_fillin(param, OP, chdata):
    Over_sample = 2
    num_files = len(chdata)
    SDDch = None
    SDDp2p = None

    for i in range(num_files):
        V = np.asarray(chdata[i].uneq_pulse_response, dtype=float)
        T = np.asarray(chdata[i].t, dtype=float)
        dt = T[1] - T[0]
        df = 1.0 / T[-1]
        f_nyq = 1.0 / dt
        f = np.arange(0, f_nyq + df / 2, df)

        chdata[i].faxis = f
        chdata[i].fmaxi = len(f)

        UI = param.ui
        M = param.samples_per_ui

        H_bt = _Bessel_Thomson_Filter(param, f, OP.Bessel_Thomson)
        H_bw = _Butterworth_Filter(param, f, OP.Butterworth)
        H_ftr = (H_bw * H_bt).ravel()

        prr = np.sinc(f * UI)  # normalised sinc = sin(pi*f*UI)/(pi*f*UI)
        prr = prr.ravel()

        fd = np.fft.fft(V)
        fd = fd[:len(fd) // 2]

        M_v = 200
        shift_vec = np.tile(np.concatenate([[1.0], np.zeros(M - 1)]), M_v)
        step_response = lfilter(V, [1.0], shift_vec)
        Vf = step_response[-1]

        f75_arr = np.where(f >= param.fb * 0.75)[0]
        f75 = int(f75_arr[0]) if len(f75_arr) > 0 else len(f)

        IL_conv = fd[:f75] / (Vf * M * Over_sample) / prr[:f75] / H_ftr[:f75]

        IL_fields = ['sdd12_raw', 'sdd21_raw', 'sdd12_orig', 'sdd21_orig',
                     'sdd12', 'sdd21', 'sdd21p', 'sdd21f']
        Zero_fields = ['sdd22_raw', 'sdd11_raw', 'sdc12_raw', 'sdc21_raw', 'sdc22_raw', 'sdc11_raw',
                       'sdd11_orig', 'sdd22_orig', 'sdd11', 'sdd22', 'sdc12', 'sdc21',
                       'sdc11', 'sdc22', 'sdc21p']
        zero_vec = np.zeros(len(IL_conv), dtype=complex)

        for field in IL_fields:
            setattr(chdata[i], field, IL_conv)
        for field in Zero_fields:
            setattr(chdata[i], field, zero_vec)

        if i == 0:
            SDDch = np.zeros((len(IL_conv), 2, 2), dtype=complex)
            SDDch[:, 0, 1] = chdata[i].sdd12_raw
            SDDch[:, 1, 0] = chdata[i].sdd21_raw
            SDDch[:, 0, 0] = chdata[i].sdd11_raw
            SDDch[:, 1, 1] = chdata[i].sdd22_raw
            SDDp2p = np.zeros(len(IL_conv), dtype=complex)

        chdata[i].TX_RL = []
        chdata[i].TDR11 = []
        chdata[i].PDTR11 = []
        chdata[i].TDR22 = []
        chdata[i].PDTR22 = []

    return chdata, param, SDDch, SDDp2p
