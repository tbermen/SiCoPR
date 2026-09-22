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
from com_functions.fn.Bessel_Thomson_Filter.py_impl import Bessel_Thomson_Filter as _Bessel_Thomson_Filter
from com_functions.fn.Butterworth_Filter.py_impl import Butterworth_Filter as _Butterworth_Filter
from com_functions.fn.bessel.py_impl import bessel as _bessel
from scipy.signal import lfilter


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

        # MATLAB builds STEP = timeseries(step_response(1:200*M), T(1:200*M)).
        # STEP is dead -- nothing reads it -- but it is evaluated, so a time
        # axis shorter than 200*samples_per_ui stops the reference before
        # IL_conv exists.  COM Octave, M=8 with 1599 samples:
        #   "error: T(1600): out of bound 1599 (dimensions are 1x1599)";
        # 1600 samples answers.  Dropping the line let this fill in fields the
        # reference declines to produce.
        if len(T) < M_v * M:
            raise IndexError(
                'TD_FD_fillin: channel %d has %d time samples; the reference '
                'indexes T(1:%d)' % (i + 1, len(T), M_v * M))

        # MATLAB f75 is 1-based, so fd(1:f75) keeps the first bin AT OR ABOVE
        # 0.75*fb as well as everything below it.  COM Octave, 2048 samples at
        # M=8 and fb=25 GHz: f(193) = 18759159745.969715 is the first bin past
        # 18.75 GHz and length(IL_conv) is 193.  Slicing to the 0-based index
        # gave 192 bins, one short, and so a short sdd21 and SDDch as well.
        f75_arr = np.where(f >= param.fb * 0.75)[0]
        # MATLAB has no empty guard: find returns [], fd(1:[]) is empty, and
        # IL_conv comes out empty.  (Unreachable while f runs to fb*M.)
        n75 = int(f75_arr[0]) + 1 if len(f75_arr) > 0 else 0

        IL_conv = fd[:n75] / (Vf * M * Over_sample) / prr[:n75] / H_ftr[:n75]

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
