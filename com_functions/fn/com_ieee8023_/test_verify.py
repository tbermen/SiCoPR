# ============================================================
# MATLAB GROUND TRUTH
# com_ieee8023_: main COM computation entry point.
# MATLAB lines 1–904.
#
# Key invariants:
# 1. sigma_X = sqrt((L^2-1)/(3*(L-1)^2))  for L=param.levels
# 2. COM = 20*log10(A_s/A_ni) when T_O=0, MLSE=0
# 3. VEC = -20*log10((A_s-A_ni)/A_s) when T_O=0
# 4. COM = 20*log10(2*A_s/A_ni) when T_O != 0, MLSE=0
# 5. ERL_ONLY → returns results after TDR, skips COM computation
# 6. eq_failed=True → returns immediately from optimize_fom result
# 7. delta_y = min(A_s/1000, OP.BinSize)  (unless force_pdf_bin_size)
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace

from com_functions.fn.com_ieee8023_.py_impl import com_ieee8023_


# ---------------------------------------------------------------------------
# Minimal param/OP/chdata helpers
# ---------------------------------------------------------------------------

def _make_param(levels=4, fb=53.125e9, T_O=0, Min_VEO_Test=0):
    N = 32
    f = np.linspace(0, fb / 2, N)
    return SimpleNamespace(
        fb=fb, levels=levels, f_r=0.75 * fb / 2,
        samples_per_ui=4, T_O=T_O, Min_VEO_Test=Min_VEO_Test,
        pass_threshold=3.0, specBER=1e-4,
        SNDR=np.array([30.0]),
        a_thru=np.array([1.0]), a_fext=np.array([0.5]), a_next=np.array([0.3]),
        a_icn_fext=0.5, a_icn_next=0.3,
        f1=float(f[5]), f2=float(f[-5]), f2_ild=float(f[-8]),
        ndfe=4, N_bmax=4, Floating_DFE=False,
        sigma_X=1.0,  # will be recomputed
        FLAG=SimpleNamespace(S2P=0),
        snpPortsOrder=np.array([1, 3, 2, 4]),
        z_p_tx_cases=np.zeros((1, 2)), z_p_next_cases=np.zeros((1, 2)),
        z_p_fext_cases=np.zeros((1, 2)), z_p_rx_cases=np.zeros((1, 2)),
        AC_CM_RMS=np.array([0.0]), PKG_Tx_FFE_preset=0,
        Pkg_Zc=np.zeros((1, 2)), pkg_Z_c=np.zeros((1, 2)),
        sigma_ns=0.0,
        samples_for_C2M=16,
        use_bmax=np.zeros(4), use_bmin=np.zeros(4),
        current_ffegain=1.0, delta_y=0.001,
        number_of_s4p_files=1,
        ui=1.0 / fb,
        sample_dt=1.0 / (fb * 4),
        package_testcase_i=1,
    )


def _make_op(ERL_ONLY=False, T_O_nonzero=False):
    return SimpleNamespace(
        ERL_ONLY=ERL_ONLY, DO_NOT_COMPUTE_COM=False, MLSE=0,
        EW=0, TDMODE=False, RX_CALIBRATION=False, PSDRXCAL=False,
        COMPUTE_COM=False, FFE_OPT_METHOD='sweep', RxFFE=False,
        DISPLAY_WINDOW=False, DEBUG=False, BREAD_CRUMBS=False,
        SAVE_TD=False, CSV_REPORT=False, WRITE_CSV_TRANSPOSED=False,
        WC_PORTZ=False, SNDR_REF=False, force_pdf_bin_size=False,
        BinSize=1e-3, pkg_len_select=np.array([1]),
        TIMESTAMP=False,
    )


def _make_chdata(N=32, fb=53.125e9):
    f = np.linspace(0, fb / 2, N)
    mag = np.ones(N, dtype=complex) * 0.5
    ch = SimpleNamespace(
        type='THRU', faxis=f, sdd21f=mag.copy(), sdd21=mag.copy(),
        sdd21_orig=mag.copy(), sdd21_raw=mag.copy(),
        sdd21p=mag.copy(), sdd21p_nodie=mag.copy(),
        scd21_orig=np.zeros(N, dtype=complex),
        sdc21_orig=np.zeros(N, dtype=complex),
        ftr=0.75 * fb / 2, base='test', ext='.s4p',
    )
    return [ch]


def _make_fom_result(A_s=0.5, eq_failed=False):
    return SimpleNamespace(
        eq_failed=eq_failed, A_s=A_s,
        best_bmax=np.zeros(4), best_bmin=np.zeros(4),
        best_current_ffegain=1.0, ctle=1, best_G_high_pass=1,
        t_s=8, txffe=np.array([1.0]),
        sbr=np.zeros(60), t=np.linspace(0, 1e-9, 60),
        IR=np.zeros(60),
    )


def _make_stubs(A_ni=0.1, A_s=0.5, eq_failed=False):
    def fd_processing(chdata, output_args, param, OP, SDDp2p, DO_ONCE):
        return chdata, output_args

    def com_fd_to_td(chdata, param, OP):
        for ch in chdata:
            ch.VCM_CD_HF_struct = SimpleNamespace(CMn=0.001)
            ch.VCM_DC_HF_struct = SimpleNamespace(CMn=0.001)
            ch.SCMR_CD_ch = 30.0
            ch.SCMR_CD_ch_pk = 32.0
            ch.SCMR_DC_ch = 28.0
            ch.SCMR_DC_ch_pk = 30.0
            ch.P_signal = A_s**2
        return chdata

    def optimize_fom(OP, param, chdata, sigma_bn, do_C2M):
        return _make_fom_result(A_s=A_s, eq_failed=eq_failed)

    def apply_eq(param, fom_result, chdata, OP):
        for ch in chdata:
            ch.eq_pulse_response = np.zeros(60)
            ch.uneq_pulse_response = np.zeros(60)
        return chdata

    def get_pdf(ch, delta_y, t_s, param, OP, iphase):
        x = np.linspace(-1, 1, 50)
        y = np.ones(50) / 50.0
        return SimpleNamespace(x=x, y=y, BinSize=x[1]-x[0])

    def create_noise_pdf(A_s_in, param, fom_result, chdata, OP, sigma_bn, PSD_results):
        # Insert -A_ni exactly so searchsorted lands on it precisely
        left = np.linspace(-A_s_in * 1.5, -A_ni - 1e-9, 50)
        right = np.linspace(-A_ni + 1e-9, A_s_in * 1.5, 50)
        x = np.concatenate([left, np.array([-A_ni]), right])
        cdf = np.zeros(len(x))
        crossing_idx = int(np.searchsorted(x, -A_ni))  # exact index of -A_ni
        cdf[:crossing_idx] = float(param.specBER) / 2.0
        cdf[crossing_idx:] = float(param.specBER) * 2.0
        pdf = SimpleNamespace(x=x, y=np.diff(np.append(cdf, cdf[-1])))
        noise_struct = SimpleNamespace(sigma_hp=0.001)
        return pdf, cdf, noise_struct

    def tdr_erl(output_args, OP, ptc_i, chdata, param):
        return output_args, [float('inf'), float('inf')], float('inf')

    def output_arg_fill(output_args, sigma_bn, Noise_Struct, COM_SNR_Struct, param, chdata, fom_result, OP):
        output_args.COM = COM_SNR_Struct.COM
        output_args.VEC_dB = COM_SNR_Struct.VEC_dB
        output_args.VEO_mV = COM_SNR_Struct.VEO_mV
        output_args.A_s = COM_SNR_Struct.A_s
        output_args.A_ni = COM_SNR_Struct.A_ni
        return output_args

    return dict(
        _FD_Processing_fn=fd_processing,
        _COM_FD_to_TD_fn=com_fd_to_td,
        _optimize_fom_fn=optimize_fom,
        _Apply_EQ_fn=apply_eq,
        _get_pdf_fn=get_pdf,
        _Create_Noise_PDF_fn=create_noise_pdf,
        _TDR_ERL_Processing_fn=tdr_erl,
        _Output_Arg_Fill_fn=output_arg_fill,
    )


# ---------------------------------------------------------------------------
# Test 1 – sigma_X derived correctly for PAM4 (levels=4)
# ---------------------------------------------------------------------------

def test_sigma_x_pam4():
    """sigma_X = sqrt((16-1)/(3*9)) = sqrt(15/27) = sqrt(5/9) for L=4."""
    param = _make_param(levels=4)
    OP = _make_op()
    stubs = _make_stubs(A_s=0.5, A_ni=0.1)

    com_ieee8023_(param, OP, _make_chdata(), **stubs)

    expected = float(np.sqrt((4**2 - 1) / (3.0 * (4 - 1)**2)))
    assert param.sigma_X == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 2 – COM formula (T_O=0, MLSE=0)
# ---------------------------------------------------------------------------

def test_com_formula_T_O_zero():
    """COM = 20*log10(A_s/A_ni) when T_O=0 and MLSE=0."""
    A_s = 0.5
    A_ni = 0.1
    param = _make_param(T_O=0)
    OP = _make_op()
    stubs = _make_stubs(A_s=A_s, A_ni=A_ni)

    result = com_ieee8023_(param, OP, _make_chdata(), **stubs)

    expected_COM = 20.0 * np.log10(A_s / A_ni)
    assert result.COM == pytest.approx(expected_COM, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 3 – VEC formula (T_O=0, MLSE=0)
# ---------------------------------------------------------------------------

def test_vec_formula_T_O_zero():
    """VEC = -20*log10((A_s - A_ni)/A_s) when T_O=0."""
    A_s = 0.5
    A_ni = 0.1
    param = _make_param(T_O=0)
    OP = _make_op()
    stubs = _make_stubs(A_s=A_s, A_ni=A_ni)

    result = com_ieee8023_(param, OP, _make_chdata(), **stubs)

    expected_VEC = -20.0 * np.log10((A_s - A_ni) / A_s)
    assert result.VEC_dB == pytest.approx(expected_VEC, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 4 – delta_y = min(A_s/1000, BinSize)
# ---------------------------------------------------------------------------

def test_delta_y_assignment():
    """delta_y = min(A_s/1000, BinSize) unless force_pdf_bin_size."""
    A_s = 0.5
    param = _make_param()
    OP = _make_op()
    OP.BinSize = 1e-3
    OP.force_pdf_bin_size = False
    stubs = _make_stubs(A_s=A_s, A_ni=0.1)

    com_ieee8023_(param, OP, _make_chdata(), **stubs)

    expected_dy = min(A_s / 1000.0, 1e-3)
    assert param.delta_y == pytest.approx(expected_dy, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 5 – ERL_ONLY → returns early without calling optimize_fom
# ---------------------------------------------------------------------------

def test_erl_only_skips_com():
    """ERL_ONLY=True → optimize_fom is never called; result has no COM field."""
    param = _make_param()
    OP = _make_op(ERL_ONLY=True)
    stubs = _make_stubs()

    optimize_called = [False]

    def tracking_optimize(OP_, param_, chdata, sigma_bn, do_C2M):
        optimize_called[0] = True
        return _make_fom_result()

    stubs['_optimize_fom_fn'] = tracking_optimize

    com_ieee8023_(param, OP, _make_chdata(), **stubs)

    assert not optimize_called[0], "optimize_fom should not be called in ERL_ONLY mode"


# ---------------------------------------------------------------------------
# Test 6 – EQ failure → function returns immediately
# ---------------------------------------------------------------------------

def test_eq_failure_returns_early():
    """fom_result.eq_failed=True → function returns before Output_Arg_Fill."""
    param = _make_param()
    OP = _make_op()
    stubs = _make_stubs(eq_failed=True)

    fill_called = [False]

    def tracking_fill(output_args, sigma_bn, Noise_Struct, COM_SNR_Struct, param, chdata, fom_result, OP):
        fill_called[0] = True
        return output_args

    stubs['_Output_Arg_Fill_fn'] = tracking_fill

    result = com_ieee8023_(param, OP, _make_chdata(), **stubs)

    assert not fill_called[0], "Output_Arg_Fill should not be called after EQ failure"
