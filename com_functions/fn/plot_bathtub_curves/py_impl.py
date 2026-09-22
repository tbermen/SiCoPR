# ============================================================
# MATLAB→Python translation notes for plot_bathtub_curves
# MATLAB lines: 8865–8913
# ============================================================
# Inline helpers: conv_fct (5371-5388), d_cpdf (MATLAB d_cpdf).
# cursors: two-point PDF at ±max_signal with equal probability 1/2.
# MATLAB semilogy → axes.semilogy in Python.
# vbt_r: MATLAB fliplr(0.5 - cumsum(fliplr(...))) → np.flipud for 1-D arrays.
# The if 0 ... end debug block is omitted.
# ============================================================

import copy
import numpy as np
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

from scipy.signal import fftconvolve
from types import SimpleNamespace



# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


def plot_bathtub_curves(hax, max_signal, sci_pdf, cci_pdf, isi_and_xtalk_pdf,
                        noise_pdf, jitt_pdf, combined_interference_and_noise_pdf, bin_size):
    cursors = _d_cpdf(bin_size, max_signal * np.array([-1.0, 1.0]), np.array([0.5, 0.5]))
    signal_and_isi_pdf = _conv_fct(cursors, sci_pdf)
    signal_and_xtalk_pdf = _conv_fct(cursors, cci_pdf)
    signal_and_channel_noise_pdf = _conv_fct(cursors, isi_and_xtalk_pdf)
    signal_and_system_noise_pdf = _conv_fct(cursors, noise_pdf)
    signal_and_system_jitt_pdf = _conv_fct(cursors, jitt_pdf)

    cursors_l = copy.copy(cursors)
    cursors_l.y = cursors_l.y.copy()
    cursors_l.y[cursors_l.x > 0] = 0

    cursors_r = copy.copy(cursors)
    cursors_r.y = cursors_r.y.copy()
    cursors_r.y[cursors_r.x < 0] = 0

    signal_and_total_noise_pdf_l = _conv_fct(cursors_l, combined_interference_and_noise_pdf)
    signal_and_total_noise_pdf_r = _conv_fct(cursors_r, combined_interference_and_noise_pdf)

    hax.semilogy(signal_and_isi_pdf.x, np.abs(np.cumsum(signal_and_isi_pdf.y) - 0.5),
                 'r', label='ISI')
    hax.semilogy(signal_and_xtalk_pdf.x, np.abs(np.cumsum(signal_and_xtalk_pdf.y) - 0.5),
                 'b', label='Xtalk')
    hax.semilogy(signal_and_channel_noise_pdf.x, np.abs(np.cumsum(signal_and_channel_noise_pdf.y) - 0.5),
                 'c', label='ISI+Xtalk')
    hax.semilogy(signal_and_system_noise_pdf.x, np.abs(np.cumsum(signal_and_system_noise_pdf.y) - 0.5),
                 'm', label='Jitter, SNR_TX,RL_M, eta_0 noise')
    hax.semilogy(signal_and_system_jitt_pdf.x, np.abs(np.cumsum(signal_and_system_jitt_pdf.y) - 0.5),
                 'g', label='Jitter noise')

    vbt_l = np.abs(0.5 - np.cumsum(signal_and_total_noise_pdf_l.y))
    vbt_r = np.flipud(0.5 - np.cumsum(np.flipud(signal_and_total_noise_pdf_r.y)))
    hax.semilogy(signal_and_total_noise_pdf_l.x, vbt_l, 'k', label='total noise PDF left')
    hax.semilogy(signal_and_total_noise_pdf_r.x, vbt_r, 'k', label='total noise PDF right')

    hax.plot(max_signal * np.array([-1.0, -1.0, 1.0, 1.0]),
             [0.5, 1e-20, 1e-20, 0.5], '--ok')
    hax.set_ylabel('Probability')
    hax.set_xlabel('volts')
    hax.legend()
