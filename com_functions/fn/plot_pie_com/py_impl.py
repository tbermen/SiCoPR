# ============================================================
# MATLAB→Python translation notes for plot_pie_com
# MATLAB lines: 9011–9091
# ============================================================
# No return value in MATLAB (void function).
# All matplotlib bar/plot calls skipped (no DISPLAY_WINDOW).
# Computes COM_per_noise breakdown and prints it.
# iex.*: find first abs(cumsum(y)) >= BER (0-based in Python: argmax of cumsum >= BER).
# maxn[0]=sci, maxn[1]=noise, maxn[2]=cci.
# COM_per_noise = COM * (maxn^2 / sum(maxn^2)).
# pfctr/mfctr: exp(+/-0.09054*delta_dB).
# Returns None (void).
# ============================================================

import numpy as np


def plot_pie_com(hax, max_signal, sci_pdf, cci_pdf, isi_and_xtalk_pdf, noise_pdf,
                 combined_interference_and_noise_pdf, bin_size, param):
    """COM breakdown computation and (skipped) pie/bar plot (MATLAB lines 9011-9091).

    Prints COM breakdown percentages. No return value.
    """
    BER = float(param.specBER)
    delta_dB = float(param.delta_IL)

    def _find_iex(pdf):
        cs = np.cumsum(np.asarray(pdf.y, dtype=float))
        idx = np.where(np.abs(cs) >= BER)[0]
        return int(idx[0]) if len(idx) > 0 else len(cs) - 1

    iex_combined = _find_iex(combined_interference_and_noise_pdf)
    iex_noise = _find_iex(noise_pdf)
    iex_cci = _find_iex(cci_pdf)
    iex_sci = _find_iex(sci_pdf)

    x_combined = np.asarray(combined_interference_and_noise_pdf.x, dtype=float)
    x_noise = np.asarray(noise_pdf.x, dtype=float)
    x_cci = np.asarray(cci_pdf.x, dtype=float)
    x_sci = np.asarray(sci_pdf.x, dtype=float)

    maxn = np.array([
        abs(x_sci[iex_sci]),
        abs(x_noise[iex_noise]),
        abs(x_cci[iex_cci]),
    ], dtype=float)
    maxn_tot = abs(x_combined[iex_combined])

    COM = 20.0 * np.log10(float(max_signal) / maxn_tot) if maxn_tot > 0 else -np.inf

    if COM < 0:
        return

    sum_maxn2 = float(np.dot(maxn, maxn))
    COM_per_noise = COM * (maxn ** 2 / sum_maxn2) if sum_maxn2 > 0 else np.zeros(3)

    COM_ISI = f'{100 * COM_per_noise[0] / np.sum(COM_per_noise):.2g}%' if np.sum(COM_per_noise) > 0 else '0%'
    COM_SYS = f'{100 * COM_per_noise[1] / np.sum(COM_per_noise):.2g}%' if np.sum(COM_per_noise) > 0 else '0%'
    COM_xtalk = f'{100 * COM_per_noise[2] / np.sum(COM_per_noise):.2g}%' if np.sum(COM_per_noise) > 0 else '0%'

    pfctr = float(np.exp(-0.09054 * delta_dB))
    mfctr = float(np.exp(0.09054 * delta_dB))

    plus_maxn = np.array([maxn[0] * pfctr, maxn[1], maxn[2] * pfctr])
    minus_maxn = np.array([maxn[0] * mfctr, maxn[1], maxn[2] * mfctr])

    plus_maxn_tot = float(np.linalg.norm(plus_maxn))
    minus_maxn_tot = float(np.linalg.norm(minus_maxn))

    COMp = 20.0 * np.log10(float(max_signal) * pfctr / maxn_tot) if maxn_tot > 0 else -np.inf
    COMm = 20.0 * np.log10(float(max_signal) * mfctr / maxn_tot) if maxn_tot > 0 else -np.inf

    print(f'ISI: {COM_ISI}, System noise: {COM_SYS}, Crosstalk: {COM_xtalk}')
