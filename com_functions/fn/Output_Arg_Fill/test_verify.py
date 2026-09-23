"""Verification tests for Output_Arg_Fill().

# ============================================================
# MATLAB GROUND TRUTH (lines 3976-4173)
# Fills output_args with COM/noise/equalization results.
# VMA: OP.TDECQ=False → VMA=[]; 'vma' → calls vma helper.
# DFE4_RSS = norm(taps[3:]); DFE2_RSS = norm(taps[1:]).
# sgm_* fields computed via pdf2sgm.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.Output_Arg_Fill.py_impl import Output_Arg_Fill


def _gaussian_pdf(sigma=0.05, bin_size=0.001):
    x = np.arange(-round(5*sigma/bin_size), round(5*sigma/bin_size)+1)*bin_size
    y = np.exp(-x**2/(2*sigma**2)); y /= y.sum()
    return SimpleNamespace(BinSize=bin_size, Min=int(round(x[0]/bin_size)), y=y, x=x)


def _dummy_pdf(bin_size=0.001):
    return SimpleNamespace(BinSize=bin_size, Min=0, y=np.array([1.0]), x=np.array([0.0]))


def _make_structs():
    bs = 0.001
    g = _gaussian_pdf(bin_size=bs)
    d = _dummy_pdf(bs)
    COM = SimpleNamespace(
        A_s=0.5, A_ni=0.3, COM=3.0, VEC_dB=-5.0, VEO_mV=50.0,
        combined_interference_and_noise_pdf=g,
        threshold_DER=1e-6,
    )
    Noise = SimpleNamespace(
        sigma_N=0.02, sigma_G=0.01, sigma_rjit=0.005, sigma_TX=0.03,
        sci_pdf=d, cci_pdf=d, isi_and_xtalk_pdf=g, noise_pdf=g, p_DD=d,
        gaussian_noise_pdf=g, jitt_pdf=d,
        peak_interference_at_BER=0.3,
        thru_peak_interference_at_BER=0.1,
        crosstalk_peak_interference_at_BER=0.2,
        MDNEXT_peak_interference=0.05,
        MDFEXT_peak_interference=0.05,
        sci_sigma=0.02, cci_sigma=0.01,
    )
    param = SimpleNamespace(
        samples_per_ui=8, fb=25e9,
        N_v=10, N_qb=0,
        specBER=1e-6, pass_threshold=3.0,
        CTLE_type='CL93',
        ctle_gdc_values=np.array([-6.0, 0.0]),
        CTLE_fp1=np.array([10e9]), CTLE_fp2=np.array([20e9]),
        CTLE_fz=np.array([5e9]),
        g_DC_HP_values=np.array([0.0]),
        f_HP=np.array([1e9]),
        f_HP_Z=np.array([1e9]), f_HP_P=np.array([2e9]),
        R_diepad=50.0, C_diepad=0.1e-12, L_comp=0.5e-9, C_bump=0.2e-12,
        levels=4, Pkg_len_TX=0.1, Pkg_len_NEXT=0.05, Pkg_len_FEXT=0.05,
        Pkg_len_RX=0.1, pkg_Z_c=50.0, C_v=0.2e-12,
        num_next=0, num_fext=0,
        T_O=0.5,
        AC_CM_RMS=np.array([0.0]),
        Floating_DFE=False, Floating_RXFFE=False,
        ndfe=3,
    )
    OP = SimpleNamespace(
        TDECQ=False,
        RX_CALIBRATION=0, PSDRXCAL=0,
        EW=0, MLSE=False, TDMODE=0,
        RxFFE=False, nburst=0,
        COM_EP_margin=3.0,
        use_simple_EP_model=True,
        PHY='C2C',
    )
    fom = SimpleNamespace(
        FOM=10.0, ctle=1, best_G_high_pass=1,
        DFE_taps=np.array([0.3, 0.2, 0.1]),
        txffe=np.array([0.0, 1.0, 0.0]),
        sbr=np.zeros(200),
        A_f=0.4, Pmax_by_Vf=1.1,
        SNR_ISI=20.0, Tr_measured_from_step=15e-12,
        tail_RSS=0.0, itick=1,
    )
    chdata = [SimpleNamespace(
        base='test_channel',
        uneq_pulse_response=np.zeros(100),
        uneq_imp_response=np.zeros(100),
        eq_pulse_response=np.zeros(100),
        t=np.linspace(0, 4e-9, 100),
    )]
    return COM, Noise, param, OP, fom, chdata


def test_FOM_set():
    """output_args.FOM equals fom_result.FOM."""
    COM, Noise, param, OP, fom, chdata = _make_structs()
    out = Output_Arg_Fill(SimpleNamespace(), None, Noise, COM, param, chdata, fom, OP)
    assert out.FOM == pytest.approx(10.0)


def test_VMA_empty_when_false():
    """output_args.VMA = [] when TDECQ=False."""
    COM, Noise, param, OP, fom, chdata = _make_structs()
    out = Output_Arg_Fill(SimpleNamespace(), None, Noise, COM, param, chdata, fom, OP)
    assert out.VMA == [] or out.VMA is None or len(out.VMA) == 0


def test_DFE4_RSS():
    """DFE4_RSS = norm(taps[3:])."""
    COM, Noise, param, OP, fom, chdata = _make_structs()
    taps = np.array([0.3, 0.2, 0.1])
    fom.DFE_taps = taps
    out = Output_Arg_Fill(SimpleNamespace(), None, Noise, COM, param, chdata, fom, OP)
    assert out.DFE4_RSS == pytest.approx(float(np.linalg.norm(taps[3:])))


def test_sgm_isi_nonnegative():
    """sgm_isi >= 0."""
    COM, Noise, param, OP, fom, chdata = _make_structs()
    out = Output_Arg_Fill(SimpleNamespace(), None, Noise, COM, param, chdata, fom, OP)
    assert out.sgm_isi >= 0


def test_file_names_quoted():
    """file_names is a quoted string."""
    COM, Noise, param, OP, fom, chdata = _make_structs()
    out = Output_Arg_Fill(SimpleNamespace(), None, Noise, COM, param, chdata, fom, OP)
    assert out.file_names.startswith('"') and out.file_names.endswith('"')


def test_COM_dB_set():
    """output_args.COM_dB == COM."""
    COM, Noise, param, OP, fom, chdata = _make_structs()
    out = Output_Arg_Fill(SimpleNamespace(), None, Noise, COM, param, chdata, fom, OP)
    assert out.COM_dB == COM.COM


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, Output_Arg_Fill from the
# 4p16p0 compat file; str2csv / pdf2sgm / Burst_Probability_Calc /
# get_pdf_from_sampled_signal / conv_fct / d_cpdf / Init_PDF_Fast taken from
# matlab/com_ieee8023_4p16p0.m).  Generated 2026-09-22.
#
# The whole function was driven under Octave on the fixture below and all 75
# output fields compared, in order, against the Python result.
# ============================================================

_N_ORACLE = 40
_T_ORACLE = np.arange(_N_ORACLE) * 1e-12
_UNEQ_IR = np.exp(-((np.arange(_N_ORACLE) - 12.0) ** 2) / 8.0)
_EQ_PR = np.concatenate([np.linspace(0, 1, 14), np.linspace(1, 0.4, _N_ORACLE - 14)])


def _oracle_pdf():
    x = np.linspace(-0.1, 0.1, 21)
    y = np.exp(-(x / 0.03) ** 2)
    return SimpleNamespace(BinSize=0.01, Min=float(round(x[0] / 0.01)),
                           x=x, y=y / y.sum())


def _oracle_structs():
    P = _oracle_pdf()
    P2 = _oracle_pdf()
    P2.y = np.roll(P2.y, 2) / np.roll(P2.y, 2).sum()
    OP = SimpleNamespace(
        TDECQ=False, RX_CALIBRATION=0.0, PSDRXCAL=0.0, TDMODE=0.0, nburst=0.0,
        RxFFE=0.0, MLSE=0.0, EW=1.0, COM_EP_margin=1.0,
        use_simple_EP_model=0.0, PHY='C2M')
    param = SimpleNamespace(
        samples_per_ui=8.0, fb=106.25e9, specBER=1e-5, levels=4.0,
        Pkg_len_TX=12.0, Pkg_len_NEXT=12.0, Pkg_len_FEXT=12.0, Pkg_len_RX=12.0,
        # Every echoed parameter differs entry to entry, and the CTLE lists
        # differ index to index, so a wrong element or a wrong index cannot
        # agree by luck.
        R_diepad=np.array([50.0, 55.0]), pkg_Z_c=np.array([87.5, 92.5]),
        C_v=np.array([0.1, 0.2]), C_diepad=np.array([0.15, 0.25]),
        L_comp=np.array([0.13, 0.23]), C_bump=np.array([0.3, 0.4]),
        num_next=0.0, num_fext=0.0, CTLE_type='CL120d',
        CTLE_fz=np.array([2e9, 3e9, 4e9]),
        CTLE_fp1=np.array([1e10, 1.1e10, 1.2e10]),
        CTLE_fp2=np.array([3e10, 3.1e10, 3.2e10]),
        ctle_gdc_values=np.array([-6.0, -9.0, -12.0]),
        g_DC_HP_values=np.array([0.0, -1.0]),
        f_HP=np.array([6.6e8, 6.6e8]),
        f_HP_Z=np.array([5.5e8, 6.5e8, 7.5e8]),
        f_HP_P=np.array([1.5e9, 2.5e9, 3.5e9]),
        Floating_DFE=0.0, Floating_RXFFE=0.0,
        N_v=3.0, N_qb=0.0, T_O=0.02, AC_CM_RMS=np.array([0.0, 0.0]),
        current_ffegain=1.0, pass_threshold=3.0, ndfe=4.0, delta_y=1e-4)
    chdata = [SimpleNamespace(
        base='thru_file_a', t=_T_ORACLE, uneq_imp_response=_UNEQ_IR,
        uneq_pulse_response=np.cumsum(_UNEQ_IR) * 0.01,
        eq_pulse_response=_EQ_PR, sigma_ACCM_at_tp0=0.002, CD_CM_RMS=0.003)]
    fom = SimpleNamespace(
        FOM=12.5, DFE_taps=np.array([0.2, -0.1, 0.05, 0.01]), tail_RSS=0.004,
        A_f=0.4, SNR_ISI=20.0, Pmax_by_Vf=0.8, Tr_measured_from_step=8e-12,
        ctle=2.0, best_G_high_pass=1.0, txffe=np.array([-0.02, 0.9, -0.08]),
        floating_tap_locations=np.array([20.0, 30.0]),
        RxFFE=np.array([0.1, 1.0, -0.1]), itick=-3.0, sbr=_EQ_PR.copy())
    Noise = SimpleNamespace(
        sigma_N=0.0012, peak_interference_at_BER=0.03,
        thru_peak_interference_at_BER=0.02, sci_sigma=0.004, cci_sigma=0.003,
        crosstalk_peak_interference_at_BER=0.01,
        MDNEXT_peak_interference=0.006, MDFEXT_peak_interference=0.007,
        isi_and_xtalk_pdf=P, noise_pdf=P2, p_DD=P, gaussian_noise_pdf=P2,
        sigma_G=0.001, sigma_rjit=0.0009, sigma_TX=0.0008, sci_pdf=P,
        cci_pdf=P2, sigma_Q=0.0005, sigma_before_clip=0.0006, peak_clip=0.02,
        p2ptosigma_clip=5.0)
    COM = SimpleNamespace(
        COM=3.4, A_s=0.35, A_ni=0.08, combined_interference_and_noise_pdf=P,
        VEC_dB=2.2, VEO_mV=120.0, EW_UI=0.4,
        eye_contour=np.array([1.0, 2.0, 3.0]), threshold_DER=1e-4,
        COM_orig=3.1, delta_COM=0.3, DER_DFE=1e-4, DER_MLSE=1e-5,
        VEC_dB_orig=2.4, delta_VEC=0.2)
    return COM, Noise, param, OP, fom, chdata


def _fill(COM, Noise, param, OP, fom, chdata):
    return Output_Arg_Fill(SimpleNamespace(), 0.0011, Noise, COM, param,
                           chdata, fom, OP)


def test_oracle_nominal_scalars():
    """COM Octave on the fixture above:
        SNR_ISI_XTK_normalized_1_sigma = 33.937094096099386
        Pre2Pmax                       = 0.022222222222222223
        uneq_FIR_peak_time             = 1.2e-11
        steady_state_voltage_weq_mV    = 3047
        DFE4_RSS                       = 0.01
        file_names                     = "thru_file_a"
        CTLE_zero_poles                = [3e9 3.1e10 1.1e10]
        HP_poles_zero                  = 6.6e8
    """
    out = _fill(*_oracle_structs())
    assert out.SNR_ISI_XTK_normalized_1_sigma == pytest.approx(
        33.937094096099386, rel=1e-12)
    assert out.Pre2Pmax == pytest.approx(0.022222222222222223, rel=1e-15)
    assert out.uneq_FIR_peak_time == pytest.approx(1.2e-11, rel=1e-15)
    assert out.steady_state_voltage_weq_mV == pytest.approx(3047.0, rel=1e-13)
    assert out.DFE4_RSS == pytest.approx(0.01, rel=1e-15)
    assert out.file_names == '"thru_file_a"'
    np.testing.assert_allclose(out.CTLE_zero_poles, [3e9, 3.1e10, 1.1e10])
    assert out.HP_poles_zero == pytest.approx(6.6e8)


def test_snr_is_written_even_when_degenerate():
    """MATLAB wraps the SNR block in `if 1`, so the field is always filled.

    COM Octave:
        Noise_Struct.peak_interference_at_BER = 0 -> +Inf
        COM_SNR_Struct.A_s = 0                    -> -Inf
    Python's `if A_s != 0 and peak != 0` guard returned [] for both.
    """
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    Noise.peak_interference_at_BER = 0.0
    assert _fill(COM, Noise, param, OP, fom, chdata
                 ).SNR_ISI_XTK_normalized_1_sigma == np.inf

    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    COM.A_s = 0.0
    assert _fill(COM, Noise, param, OP, fom, chdata
                 ).SNR_ISI_XTK_normalized_1_sigma == -np.inf


def test_pre2pmax_divides_by_zero_like_matlab():
    """COM Octave, fom_result.txffe=[-0.02 0 -0.08]: Pre2Pmax == Inf.
    The `if taps[-2] != 0` guard returned [] instead."""
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    fom.txffe = np.array([-0.02, 0.0, -0.08])
    assert _fill(COM, Noise, param, OP, fom, chdata).Pre2Pmax == np.inf


def test_uneq_fir_peak_time_returns_every_tie():
    """t(uneq_imp_response==max(...)) is a 1xN, not just the first hit.

    COM Octave, uneq_imp_response = [ir(1:20) ir(1:20)] (peak occurs twice):
        uneq_FIR_peak_time == [1.2e-11 3.2e-11]
    Python reported only 1.2e-11.
    """
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    chdata[0].uneq_imp_response = np.concatenate([_UNEQ_IR[:20], _UNEQ_IR[:20]])
    got = np.atleast_1d(_fill(COM, Noise, param, OP, fom, chdata
                              ).uneq_FIR_peak_time)
    np.testing.assert_allclose(got, [1.2e-11, 3.2e-11], rtol=1e-13)


def test_burst_probability_path_runs_with_float_levels():
    """OP.nburst>0 reaches Burst_Probability_Calc, which builds a PDF with
    ones(1,param.levels).  param.levels is a float and np.ones(4.0) raises
    TypeError, so every nburst>0 run died here.

    COM Octave, OP.nburst=4, OP.COM_EP_margin=-15, param.delta_y=0.01:
      error_propagation_probability =
        [0.016129577635929825 0.39850791318904072
         0.3825871863621198   0.38207597837409674]
      burst_probabilities =
        [0.016129577635929825 0.0064277643243150157
         0.0024591802674384941 0.00093959370667983549]
    """
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    OP.nburst = 4.0
    OP.COM_EP_margin = -15.0
    param.delta_y = 0.01
    out = _fill(COM, Noise, param, OP, fom, chdata)
    np.testing.assert_allclose(
        np.ravel(out.error_propagation_probability),
        [0.016129577635929825, 0.39850791318904072,
         0.3825871863621198, 0.38207597837409674], rtol=1e-12)
    np.testing.assert_allclose(
        np.ravel(out.burst_probabilities),
        [0.016129577635929825, 0.0064277643243150157,
         0.0024591802674384941, 0.00093959370667983549], rtol=1e-12)


def test_octave_cl120e_reports_five_zeros_and_poles():
    """COM Octave, the fixture above with param.CTLE_type = 'CL120e'.

    CL120e reports a five-entry list, in the MATLAB order
      [CTLE_fz(ctle) f_HP_Z(ctle) CTLE_fp2(ctle) CTLE_fp1(ctle) f_HP_P(ctle)]
    and leaves the two CL120d high-pass fields empty:
        CTLE_zero_poles = [3e9 6.5e8 3.1e10 1.1e10 2.5e9]
        CTLE_DC_gain_dB = -9
        g_DC_HP         = []
        HP_poles_zero   = []
    fom_result.ctle is 2 and best_G_high_pass is 1, and all five frequencies
    differ, so neither a permuted list nor a lookup by best_G_high_pass can
    agree by accident.  All 75 output fields were compared against Octave on
    this fixture; these are the four the branch decides.
    """
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    param.CTLE_type = 'CL120e'
    out = _fill(COM, Noise, param, OP, fom, chdata)
    np.testing.assert_allclose(out.CTLE_zero_poles,
                               [3e9, 6.5e8, 3.1e10, 1.1e10, 2.5e9], rtol=1e-15)
    assert out.CTLE_DC_gain_dB == pytest.approx(-9.0, rel=1e-15)
    assert len(np.atleast_1d(out.g_DC_HP)) == 0
    assert len(np.atleast_1d(out.HP_poles_zero)) == 0


def test_octave_termination_and_package_params_are_echoed():
    """The two field-name loops copy param straight into output_args.

    COM Octave on the fixture above returns these unchanged, as arrays, in
    the order given -- a scalarised or truncated copy is a divergence:
        R_diepad     [50 55]        C_diepad  [0.15 0.25]
        L_comp       [0.13 0.23]    C_bump    [0.3 0.4]
        levels       4              pkg_Z_c   [87.5 92.5]
        C_v          [0.1 0.2]      Pkg_len_TX/NEXT/FEXT/RX  12
    R_diepad is named in both loops; the second write must still land.
    """
    out = _fill(*_oracle_structs())
    expected = {
        'R_diepad': [50.0, 55.0],
        'C_diepad': [0.15, 0.25],
        'L_comp': [0.13, 0.23],
        'C_bump': [0.3, 0.4],
        'levels': [4.0],
        'Pkg_len_TX': [12.0],
        'Pkg_len_NEXT': [12.0],
        'Pkg_len_FEXT': [12.0],
        'Pkg_len_RX': [12.0],
        'pkg_Z_c': [87.5, 92.5],
        'C_v': [0.1, 0.2],
    }
    for name, want in expected.items():
        got = np.atleast_1d(np.asarray(getattr(out, name), dtype=float))
        assert got.shape == (len(want),), '%s: %r' % (name, got)
        np.testing.assert_allclose(got, want, rtol=1e-15)


def test_burst_probability_simple_ep_model():
    """COM Octave, same but OP.use_simple_EP_model=1 (always convolves with the
    burst-1 PDF and always takes the largest tap):
      error_propagation_probability =
        [0.016129577635929825 0.39850791318904072
         0.39850791318904072  0.39850791318904072]
      burst_probabilities =
        [0.016129577635929825 0.0064277643243150157
         0.0025615149473537414 0.0010207839762724749]
    """
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    OP.nburst = 4.0
    OP.COM_EP_margin = -15.0
    OP.use_simple_EP_model = 1.0
    param.delta_y = 0.01
    out = _fill(COM, Noise, param, OP, fom, chdata)
    np.testing.assert_allclose(
        np.ravel(out.error_propagation_probability),
        [0.016129577635929825, 0.39850791318904072,
         0.39850791318904072, 0.39850791318904072], rtol=1e-12)
    np.testing.assert_allclose(
        np.ravel(out.burst_probabilities),
        [0.016129577635929825, 0.0064277643243150157,
         0.0025615149473537414, 0.0010207839762724749], rtol=1e-12)


# --------------------------------------------------------------------------
# ML 176 guards the AC common-mode outputs with `sum(param.AC_CM_RMS) ~= 0`,
# not `> 0`. The two conditions differ on exactly one input -- a negative sum
# -- and nothing covered it, so the reference's `~= 0` silently becoming `> 0`
# was invisible. A negative RMS is not physical; the branch condition the
# reference writes is still the one the port has to carry.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('ac_cm,populated', [
    (np.array([0.0, 0.0]), False),
    (np.array([0.01, 0.0]), True),
    (np.array([-0.01, 0.0]), True),     # ~= 0 takes the branch; > 0 would not
    (np.array([0.01, -0.01]), False),   # the SUM is zero, so neither does
])
def test_ac_cm_branch_is_not_equal_zero_not_greater_than_zero(ac_cm, populated):
    COM, Noise, param, OP, fom, chdata = _oracle_structs()
    param.AC_CM_RMS = ac_cm
    out = _fill(COM, Noise, param, OP, fom, chdata)

    got_tp0 = out.sigma_ACCM_at_tp0_mV
    got_rx = out.sigma_AC_CCM_at_rxpkg_output_mV
    if populated:
        # chdata[0].sigma_ACCM_at_tp0 = 0.002 and CD_CM_RMS = 0.003, in volts
        assert got_tp0 == pytest.approx(2.0), (
            'sum(AC_CM_RMS) = %g is non-zero, so ML 177 runs; got %r'
            % (float(np.sum(ac_cm)), got_tp0))
        assert got_rx == pytest.approx(3.0)
    else:
        assert got_tp0 == [] and got_rx == [], (
            'sum(AC_CM_RMS) = %g is zero, so ML 180 runs and both come back '
            'empty; got %r and %r' % (float(np.sum(ac_cm)), got_tp0, got_rx))
