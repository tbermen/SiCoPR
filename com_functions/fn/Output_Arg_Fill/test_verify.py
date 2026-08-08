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
