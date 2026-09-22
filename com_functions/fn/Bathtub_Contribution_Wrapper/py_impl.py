# ============================================================
# MATLAB→Python translation notes for Bathtub_Contribution_Wrapper
# MATLAB lines: 977–1028
# ============================================================
# MATLAB figure/subplot GUI → matplotlib.  msgbox/movegui omitted.
# OP.COM_CONTRIBUTION_CURVES=False → plot_bathtub_curves; True → plot_pie_com (top-level,
#   resolved in the assembled module), MATLAB L1003-1021.
# MATLAB case_number is 1-based; matplotlib add_subplot uses 1-based too.
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


def _plot_bathtub_curves(hax, max_signal, sci_pdf, cci_pdf, isi_and_xtalk_pdf,
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
    hax.semilogy(signal_and_channel_noise_pdf.x,
                 np.abs(np.cumsum(signal_and_channel_noise_pdf.y) - 0.5),
                 'c', label='ISI+Xtalk')
    hax.semilogy(signal_and_system_noise_pdf.x,
                 np.abs(np.cumsum(signal_and_system_noise_pdf.y) - 0.5),
                 'm', label='Jitter, SNR_TX,RL_M, eta_0 noise')
    hax.semilogy(signal_and_system_jitt_pdf.x,
                 np.abs(np.cumsum(signal_and_system_jitt_pdf.y) - 0.5),
                 'g', label='Jitter noise')

    vbt_l = np.abs(0.5 - np.cumsum(signal_and_total_noise_pdf_l.y))
    vbt_r = np.flipud(0.5 - np.cumsum(np.flipud(signal_and_total_noise_pdf_r.y)))
    hax.semilogy(signal_and_total_noise_pdf_l.x, vbt_l, 'k', label='total noise PDF left')
    hax.semilogy(signal_and_total_noise_pdf_r.x, vbt_r, 'k', label='total noise PDF right')

    hax.plot(max_signal * np.array([-1.0, -1.0, 1.0, 1.0]), [0.5, 1e-20, 1e-20, 0.5], '--ok')
    hax.set_ylabel('Probability')
    hax.set_xlabel('volts')
    hax.legend()


def Bathtub_Contribution_Wrapper(COM_SNR_Struct, Noise_Struct, param, chdata, OP):
    case_number = int(param.package_testcase_i)  # 1-based
    if not OP.COM_CONTRIBUTION_CURVES:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig_name = 'Voltage bathtub curves'
        existing = [plt.figure(n) for n in plt.get_fignums()
                    if plt.figure(n).get_label() == fig_name]
        fig = existing[0] if existing else plt.figure(num=fig_name)
        n_cases = len(OP.pkg_len_select)
        hax = fig.add_subplot(n_cases, 1, case_number)
        _plot_bathtub_curves(
            hax,
            COM_SNR_Struct.A_s,
            Noise_Struct.sci_pdf,
            Noise_Struct.cci_pdf,
            Noise_Struct.isi_and_xtalk_pdf,
            Noise_Struct.noise_pdf,
            Noise_Struct.jitt_pdf,
            COM_SNR_Struct.combined_interference_and_noise_pdf,
            param.delta_y,
        )
        hax.set_ylim([param.specBER / 10, 1])
        xlims = hax.get_xlim()
        hax.plot(list(xlims), [param.specBER, param.specBER], 'r:')
        # MATLAB appends the space BEFORE regexprep and the format adds another,
        # so the title ends in TWO spaces.  COM Octave:
        #   sprintf('case %d VBC: %s ',1,regexprep(['Case_1_port_A',' '],'_',' '))
        #   -> 'case 1 VBC: Case 1 port A  '  (27 chars)
        base_str = (chdata[0].base + ' ').replace('_', ' ')
        hax.set_title(f'case {case_number} VBC: {base_str} ')
    else:
        # MATLAB L1003-1021: COM contribution (rough allocation) branch.
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig_name = 'COM Contributions (Rough Allocations)'
        existing = [plt.figure(n) for n in plt.get_fignums()
                    if plt.figure(n).get_label() == fig_name]
        fig = existing[0] if existing else plt.figure(num=fig_name)
        n_cases = len(OP.pkg_len_select)
        hax = fig.add_subplot(n_cases, 1, case_number)
        # top-level plot_pie_com (resolved in the assembled module); note: no jitt_pdf,
        # and it takes param (MATLAB L1011-1018).
        plot_pie_com(
            hax,
            COM_SNR_Struct.A_s,
            Noise_Struct.sci_pdf,
            Noise_Struct.cci_pdf,
            Noise_Struct.isi_and_xtalk_pdf,
            Noise_Struct.noise_pdf,
            COM_SNR_Struct.combined_interference_and_noise_pdf,
            param.delta_y,
            param,
        )
        base_str = (chdata[0].base + ' ').replace('_', ' ')   # see VBC branch
        hax.set_title(f'case {case_number} rough COM impact: {base_str} ')
