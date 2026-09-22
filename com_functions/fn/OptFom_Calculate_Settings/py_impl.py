# ============================================================
# MATLAB→Python translation notes for OptFom_Calculate_Settings
# MATLAB lines: 3034–3143
# ============================================================
# OptFom_Calc_Hr inlined as _OptFom_Calc_Hr (with filter helpers).
# OptFom_FD_or_TD_Fields inlined trivially.
# f_xc: ((1:512)-1)/512*baud_rate*M/2 → np.arange(512)/512*...
# GDC Qual matrix: ones or zeros depending on gqual/g2qual values.
# Peak search range: 0-based Python indices.
# phase_memory: shape (len(f), num_txffe_taps [+ RxFFE taps]).
# MATLAB 1-based phase_memory(:,k) with k from 1..N_taps →
#   Python phase_memory[:, k] with k from 0..N_taps-1
#   exponent: (k_matlab - cursor_index) = (k_python + 1 - cursor_index)
# scipy.signal.lfilter for non-TDMODE peak search.
# ============================================================

import math
import numpy as np

from scipy.signal import lfilter
from types import SimpleNamespace

_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]


def _bessel_poly(n):
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        a[ii] = (math.factorial(2 * n - ii)
                 / (2 ** (n - ii) * math.factorial(ii) * math.factorial(n - ii)))
    return a


def _butterworth(param, f, use_BW):
    if not use_BW:
        return np.ones(len(f))
    s = 1j * np.asarray(f) / (param.fb_BW_cutoff * param.fb)
    return 1.0 / np.polyval(_BW_POLY, s)


def _bessel_thomson(param, f, use_BT):
    if not use_BT:
        return np.ones(len(f))
    a = _bessel_poly(param.BTorder)
    acoef = a[::-1]
    s = 1j * np.asarray(f) / (param.fb_BT_cutoff * param.fb)
    return a[0] / np.polyval(acoef, s)


def _tukey_window(f, fr, fb_top):
    fperiod = 2 * (fb_top - fr)
    H = np.where(
        f < fr, 1.0,
        np.where(
            (f >= fr) & (f <= fb_top),
            0.5 * np.cos(2 * np.pi * (f - fb_top) / fperiod - np.pi) + 0.5,
            0.0,
        ),
    )
    return H[:len(f)]


def _raised_cosine(param, f, use_RC):
    if not use_RC:
        return np.ones(len(f))
    return _tukey_window(f, param.RC_Start, param.RC_end)


def _OptFom_Calc_Hr(f, param, OP):
    f = np.asarray(f, dtype=float)
    H_bt = _bessel_thomson(param, f, OP.Bessel_Thomson)
    H_bw = _butterworth(param, f, OP.Butterworth)
    H_RCos = _raised_cosine(param, f, OP.Raised_Cosine)
    return H_bw * H_bt * H_RCos


def OptFom_Calculate_Settings(txffe_matrix, chdata, param, OP):
    baud_rate = 1.0 / param.ui
    f = np.asarray(chdata[0].faxis, dtype=float)
    M = int(param.samples_per_ui)

    SETTINGS = SimpleNamespace()
    SETTINGS.min_number_of_UI_in_response = 40
    SETTINGS.delta_sbr = []

    SETTINGS.H_r = _OptFom_Calc_Hr(f, param, OP)

    N_fft_by2 = 512
    SETTINGS.N_fft_by2 = N_fft_by2
    SETTINGS.f_xc = (np.arange(N_fft_by2) / N_fft_by2
                     * baud_rate * M / 2)
    SETTINGS.H_r_xc = _OptFom_Calc_Hr(SETTINGS.f_xc, param, OP)

    if OP.USE_ETA0_PSD:
        fspike = 1e9
        SETTINGS.H_sy = np.sinc(np.sqrt(2) * (f - fspike) / fspike) ** 2
    else:
        SETTINGS.H_sy = np.ones(len(f))

    # GDC Qual matrix
    gqual = np.asarray(param.gqual) if hasattr(param, 'gqual') and param.gqual is not None else np.array([])
    g2qual = np.asarray(param.g2qual) if hasattr(param, 'g2qual') and param.g2qual is not None else np.array([])
    gdc_values = np.asarray(param.ctle_gdc_values)
    g_DC_HP_values = np.asarray(param.g_DC_HP_values)

    if param.CTLE_type != 'CL120d':
        qual = np.ones((1, len(gdc_values)))
    else:
        if gqual.size == 0 and g2qual.size == 0:
            qual = np.ones((len(g_DC_HP_values), len(gdc_values)))
        else:
            qual = np.zeros((len(g_DC_HP_values), len(gdc_values)))
            # MATLAB's sort(...,'descend') is STABLE: tied values keep their
            # original order. Reversing a stable ascending sort reverses them
            # instead. COM Octave, sort([3 1 3 2 1],'descend') -> index
            # [1 3 4 2 5]; np.argsort(a,'stable')[::-1] gives [3 1 4 5 2].
            # With g2qual = [0 -2 -2] that swapped two qual rows outright.
            si = np.argsort(-g2qual, kind='stable')
            g2qual_s = g2qual[si]
            # MATLAB's gqual(si,:) reads a bare `[0 -6]` as ONE row of two
            # columns. Indexing a 1-D array with si instead picked out single
            # elements, so a config with one qual pair compared gdc against
            # that element twice and qual came back all zeros. COM Octave,
            # gqual = [0 -6], g2qual = 0, gdc = [-1 -3 -5 -7 -9 -11]:
            # qual = [1 1 1 0 0 0].
            gqual_s = np.array(np.atleast_2d(gqual)[si, :], dtype=float)

            g2qual_pairs = np.zeros((len(g2qual_s), 2))
            for kk in range(len(g2qual_s)):
                if kk == 0:
                    g2qual_pairs[kk, :] = [g2qual_s[kk] + np.finfo(float).eps, g2qual_s[kk]]
                else:
                    g2qual_pairs[kk, :] = [g2qual_s[kk - 1], g2qual_s[kk]]
                gqual_s[kk, :] = np.sort(gqual_s[kk, :])[::-1]

            for jj in range(len(g_DC_HP_values)):
                for ii in range(len(gdc_values)):
                    for kk in range(len(g2qual_s)):
                        g2lo = g2qual_pairs[kk, 1]
                        g2hi = g2qual_pairs[kk, 0]
                        if g_DC_HP_values[jj] >= g2lo and g_DC_HP_values[jj] < g2hi:
                            # MATLAB compares against gqual(kk,2) and
                            # gqual(kk,1) of the row it just sorted descending
                            # — the two LARGEST entries, not the min and the
                            # max. They coincide for a 2-column gqual and part
                            # company beyond that. COM Octave, gqual =
                            # [0 -3 -6; -6 -9 -12], g2qual = [0 -2],
                            # gdc = [-1 -3 -5 -7 -9 -11]: qual row 1 is
                            # [1 1 0 0 0 0], where min/max gave [1 1 1 0 0 0].
                            row = gqual_s[kk, :]
                            glo = float(row[1])
                            ghi = float(row[0])
                            if gdc_values[ii] >= glo and gdc_values[ii] < ghi:
                                qual[jj, ii] = 1
                                break
    SETTINGS.qual = qual

    # Speed up search for max(sbr) by pre-computing search range
    if OP.TDMODE:
        uneq_field = 'uneq_pulse_response'
    else:
        uneq_field = 'uneq_imp_response'

    uneq_data = np.asarray(getattr(chdata[0], uneq_field), dtype=float)

    if OP.TDMODE:
        init_max = int(np.argmax(uneq_data))
    else:
        filtered = lfilter(np.ones(M), [1.0], uneq_data)
        init_max = int(np.argmax(filtered))

    UI_max_window = 20
    start_max_idx = max(0, init_max - UI_max_window * M)
    end_max_idx = min(len(uneq_data) - 1, init_max + UI_max_window * M)
    SETTINGS.start_max_idx = start_max_idx
    SETTINGS.end_max_idx = end_max_idx
    SETTINGS.Peak_Search_Range = np.arange(start_max_idx, end_max_idx + 1)

    # Phase shift exponentials (pre-computed for speed)
    num_txffe_taps = int(np.asarray(txffe_matrix).shape[1])
    phase_memory = np.zeros((len(f), num_txffe_taps), dtype=complex)
    for k in range(num_txffe_taps):  # k is 0-based; MATLAB k was 1-based
        phase_memory[:, k] = np.exp(-1j * 2 * np.pi
                                    * (k + 1 - param.cursor_index)
                                    * f / param.fb)

    if OP.RxFFE:
        cmx = int(param.RxFFE_cmx)
        cpx = int(param.RxFFE_cpx)
        n_rxffe = cmx + cpx + 1
        phase_memory_rxffe = np.zeros((len(f), n_rxffe), dtype=complex)
        for idx, ii in enumerate(range(-cmx, cpx + 1)):
            phase_memory_rxffe[:, idx] = np.exp(-1j * 2 * np.pi
                                                 * (ii + 1) * f / param.fb)
        phase_memory = np.hstack([phase_memory, phase_memory_rxffe])

    SETTINGS.phase_memory = phase_memory
    return SETTINGS
