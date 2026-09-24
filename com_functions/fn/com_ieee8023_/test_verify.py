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


# ============================================================
# Test 7 - the MMSE + RxFFE PSD block (MATLAB L537-560), the branch the
# 'S_xn' option string sits in.
#
# Behaviour-only, and honestly so: com_ieee8023_ is the top-level driver, and
# this block's numbers come entirely from get_PSDs, which cannot be run here
# without the whole engine.  What the reference fixes, and what is therefore
# asserted, is the CHOREOGRAPHY around the two get_PSDs calls:
#
#   OP.WO_TXFFE=1;                          <- first call sees 1
#   PSD_results.w    = fom_result.RxFFE;
#   PSD_results.S_rn = fom_result.PSD_results.S_rn;
#   PSD_results.S_in = fom_result.PSD_results.S_in;
#   PSD_results = get_PSDs(..., param.ctle_gdc_values(fom_result.ctle),
#                               param.g_DC_HP_values(fom_result.best_G_high_pass), ...);
#   OP.WO_TXFFE=0;                          <- second call sees 0
#   PSD_results.S_xn    = fom_result.PSD_results.S_xn;
#   PSD_results.S_tn    = fom_result.PSD_results.S_tn;
#   PSD_results.S_jn    = fom_result.PSD_results.S_jn;
#   PSD_results.S_rj_jn = fom_result.PSD_results.S_rj_jn;
#   PSD_results = get_PSDs(...same arguments...);
#   output_args.noiseRMS_mV.{rn,tn,xn,jn,in} = S_*_rms*1000;
#
# The order is the whole content of the branch: the four crosstalk/jitter
# PSDs must be seeded BETWEEN the two calls, because the first call runs
# without the Tx FFE and the second one adjusts them for the Rx FFE.  Seeding
# them before the first call, or leaving WO_TXFFE at 1 for the second, would
# still produce a PSD_results with every field present.
#
# One divergence this closes: the six output_args.noiseRMS_mV assignments
# (MATLAB L554-559) were missing from the port entirely, so the field did not
# exist on any MMSE+RxFFE run.  MATLAB writes .tn twice, at L555 and L557,
# with the same value -- harmless upstream redundancy, noted so the single
# assignment here is not mistaken for a dropped line.
# ============================================================

def _mmse_param():
    param = _make_param()
    param.ctle_gdc_values = np.array([-5.0, -8.0, -11.0, -14.0])
    param.g_DC_HP_values = np.array([0.0, -1.0, -2.0, -3.0])
    return param


def _mmse_fom_result(A_s=0.5):
    fom = _make_fom_result(A_s=A_s)
    fom.ctle = 2                 # MATLAB 1-based -> ctle_gdc_values(2) = -8
    fom.best_G_high_pass = 3     # MATLAB 1-based -> g_DC_HP_values(3) = -2
    fom.RxFFE = np.array([0.25, 1.0, -0.125])
    fom.PSD_results = SimpleNamespace(
        S_rn=np.array([1.0, 2.0]), S_in=np.array([3.0, 4.0]),
        S_xn=np.array([5.0, 6.0]), S_tn=np.array([7.0, 8.0]),
        S_jn=np.array([9.0, 10.0]), S_rj_jn=np.array([11.0, 12.0]),
        iphase=np.array([1]))
    return fom


_RMS = dict(S_rn_rms=0.0011, S_tn_rms=0.0022, S_xn_rms=0.0033,
            S_jn_rms=0.0044, S_in_rms=0.0055)


def _mmse_stubs(calls):
    fom = _mmse_fom_result()
    stubs = _make_stubs(A_s=0.5, A_ni=0.1)
    stubs['_optimize_fom_fn'] = lambda OP_, p, ch, sig, do_C2M: fom

    def get_PSDs(PSD_results, pulse, t_s, txffe, g_dc, g_hp, p, ch, OP_):
        calls.append(SimpleNamespace(
            seen=dict(vars(PSD_results)),
            WO_TXFFE=getattr(OP_, 'WO_TXFFE', None),
            g_dc=g_dc, g_hp=g_hp, t_s=t_s))
        out = SimpleNamespace(**vars(PSD_results))
        for k, v in _RMS.items():
            setattr(out, k, v)
        return out

    stubs['_get_PSDs_fn'] = get_PSDs
    return stubs, fom


def _run_mmse():
    calls = []
    param = _mmse_param()
    OP = _make_op()
    OP.FFE_OPT_METHOD = 'MMSE'
    OP.RxFFE = True
    stubs, fom = _mmse_stubs(calls)
    result = com_ieee8023_(param, OP, _make_chdata(), **stubs)
    return result, calls, fom, OP


def test_mmse_rxffe_psd_block_calls_get_PSDs_twice():
    """Two get_PSDs calls, the first with OP.WO_TXFFE=1 and the second with 0."""
    _, calls, _, OP = _run_mmse()
    assert len(calls) == 2
    assert calls[0].WO_TXFFE == 1
    assert calls[1].WO_TXFFE == 0
    assert OP.WO_TXFFE == 0, 'WO_TXFFE must be left at 0 after the block'


def test_mmse_rxffe_psd_first_call_seeded_with_w_srn_sin_only():
    """Before the first call PSD_results holds only w, S_rn and S_in."""
    _, calls, fom, _ = _run_mmse()
    assert set(calls[0].seen) == {'w', 'S_rn', 'S_in'}
    np.testing.assert_array_equal(calls[0].seen['w'], fom.RxFFE)
    np.testing.assert_array_equal(calls[0].seen['S_rn'], fom.PSD_results.S_rn)
    np.testing.assert_array_equal(calls[0].seen['S_in'], fom.PSD_results.S_in)


def test_mmse_rxffe_S_xn_seeded_between_the_two_calls():
    """S_xn, S_tn, S_jn and S_rj_jn arrive from fom_result between the calls.

    This is the 'S_xn' branch: absent from the first call, present and equal
    to fom_result.PSD_results on the second.
    """
    _, calls, fom, _ = _run_mmse()
    for fld in ('S_xn', 'S_tn', 'S_jn', 'S_rj_jn'):
        assert fld not in calls[0].seen, '%s must NOT be seeded before call 1' % fld
        assert fld in calls[1].seen, '%s must be seeded before call 2' % fld
        np.testing.assert_array_equal(calls[1].seen[fld],
                                      getattr(fom.PSD_results, fld))


def test_mmse_rxffe_ctle_gains_are_one_based_lookups():
    """ctle_gdc_values(fom_result.ctle) and g_DC_HP_values(best_G_high_pass)
    are MATLAB 1-based; ctle=2 -> -8 dB, best_G_high_pass=3 -> -2 dB."""
    _, calls, _, _ = _run_mmse()
    for c in calls:
        assert c.g_dc == -8.0
        assert c.g_hp == -2.0


def test_mmse_rxffe_noiseRMS_mV_filled():
    """output_args.noiseRMS_mV = S_*_rms*1000, in the reference's five fields.

    MATLAB L554-559.  These assignments were missing from the port, so the
    field did not exist on any MMSE+RxFFE run.
    """
    result, _, _, _ = _run_mmse()
    n = result.noiseRMS_mV
    assert n.rn == _RMS['S_rn_rms'] * 1000
    assert n.tn == _RMS['S_tn_rms'] * 1000
    assert n.xn == _RMS['S_xn_rms'] * 1000
    assert n.jn == _RMS['S_jn_rms'] * 1000
    # 'in' is a Python keyword, so the reference's field name is read by name.
    assert getattr(n, 'in') == _RMS['S_in_rms'] * 1000


def test_mmse_rxffe_block_skipped_without_RxFFE():
    """The other side of the branch: FFE_OPT_METHOD='sweep' runs no get_PSDs
    call and leaves output_args without noiseRMS_mV."""
    calls = []
    param = _mmse_param()
    OP = _make_op()                      # FFE_OPT_METHOD='sweep', RxFFE=False
    stubs, _ = _mmse_stubs(calls)
    result = com_ieee8023_(param, OP, _make_chdata(), **stubs)
    assert calls == []
    assert not hasattr(result, 'noiseRMS_mV')


# ---------------------------------------------------------------------------
# The checkpoint harness hook (tests/test_octave_checkpoints.py reads what it
# writes). Three properties, each of which the harness depends on:
#   * unset, it does nothing at all -- a normal run must not write files;
#   * set, each call is a SNAPSHOT: the pipeline mutates chdata and its kin in
#     place, and a checkpoint that recorded the later state would compare the
#     wrong stage against Octave;
#   * a stage reached twice is numbered, as com_checkpoint.m numbers it, so the
#     two get_PSDs calls do not overwrite one another.
# ---------------------------------------------------------------------------
import pickle as _pickle                                     # noqa: E402

from com_functions.fn.com_ieee8023_.py_impl import _checkpoint   # noqa: E402


def _read_ck(path):
    with open(path, 'rb') as fh:
        blob = _pickle.load(fh)
    return {k: _pickle.loads(v) for k, v in blob['values'].items()}, blob['failed']


def test_checkpoint_does_nothing_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv('COM_CHECKPOINT_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    _checkpoint('05_optimize_fom', 1, fom_result=SimpleNamespace(FOM=1.0))
    assert list(tmp_path.iterdir()) == [], 'wrote files with the hook unset'


def test_checkpoint_is_a_snapshot_and_numbers_repeats(tmp_path, monkeypatch):
    monkeypatch.setenv('COM_CHECKPOINT_DIR', str(tmp_path))
    ch = [SimpleNamespace(v=np.array([1.0, 2.0]))]
    _checkpoint('07_get_PSDs', 2, chdata=ch)
    ch[0].v[0] = 99.0                        # the pipeline mutates in place
    _checkpoint('07_get_PSDs', 2, chdata=ch)
    first, _ = _read_ck(tmp_path / '07_get_PSDs_pc2.pkl')
    second, _ = _read_ck(tmp_path / '07_get_PSDs_pc2_2.pkl')
    assert first['chdata'][0].v.tolist() == [1.0, 2.0], (
        'the first checkpoint recorded a later state')
    assert second['chdata'][0].v.tolist() == [99.0, 2.0]


def test_checkpoint_skips_what_cannot_be_pickled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('COM_CHECKPOINT_DIR', str(tmp_path))
    _checkpoint('01_read_s4p_files', 1, param=SimpleNamespace(fb=53.125e9),
                OP=SimpleNamespace(cb=lambda x: x))
    values, failed = _read_ck(tmp_path / '01_read_s4p_files_pc1.pkl')
    assert values['param'].fb == 53.125e9
    assert 'OP' in failed and 'OP' not in values
    assert 'not saved' in capsys.readouterr().err
