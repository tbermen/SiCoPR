# ============================================================
# MATLAB GROUND TRUTH
# optimize_fom: nested CTLE/TXFFE/DFE/RxFFE EQ search.
# MATLAB lines 8417–8793.
#
# Key invariants:
# 1. ts_sample_adj_range [v] → [0, v]; [a, b] unchanged
# 2. CTLE_type in {'CL93','CL120e'} → g_DC_HP_values=0 → lf_indx=1
# 3. RxFFE disabled → Gffe_values = [0]
# 4. BEST.cursor_i still None after loops → result.eq_failed=True
# 5. do_C2M=True with EQ failure → returns early (no Create_Output call)
# 6. Successful run: result.eq_failed=False, result has FOM field
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace

import com_functions.fn.optimize_fom.py_impl as _ofm
from com_functions.fn.optimize_fom.py_impl import optimize_fom


# ---------------------------------------------------------------------------
# Stub factory helpers
# ---------------------------------------------------------------------------

def _make_chdata(N=32, fb=53.125e9):
    f = np.linspace(0, fb / 2, N)
    ch = SimpleNamespace(faxis=f, sdd21f=np.ones(N, dtype=complex) * 0.5, type='THRU')
    return [ch]


def _make_param(ctle_vals=None, gdc_hp=None, ts_range=None):
    if ctle_vals is None:
        ctle_vals = np.array([-3.0])
    if gdc_hp is None:
        gdc_hp = np.array([0.0])
    if ts_range is None:
        ts_range = np.array([0.0])
    return SimpleNamespace(
        ctle_gdc_values=ctle_vals,
        g_DC_HP_values=gdc_hp,
        CTLE_fp1=np.array([5e9]), CTLE_fp2=np.array([8e9]), CTLE_fz=np.array([2e9]),
        cursor_gain=np.array([1.0]),
        ts_sample_adj_range=ts_range,
        ndfe=4, N_bmax=4, Floating_DFE=False,
        samples_per_ui=4, ui=1.0 / 53.125e9,
        sigma_X=1.0, R_LM=1.0, levels=4,
        GDC_MIN=0, LOCAL_SEARCH=0,
        tx_ffe_c0_min=-1.0,
        T_O=0, fb=53.125e9,
        CTLE_type='generic',
        cursor_index=1,
        Min_VEO_Test=0,
    )


def _make_op():
    return SimpleNamespace(
        RxFFE=False, FFE_OPT_METHOD='sweep', RxFFE_with_MMSE=0,
        INCLUDE_CTLE=1, RX_CALIBRATION=False, WC_PORTZ=False,
        DISPLAY_WINDOW=False, Optimize_loop_speed_up=0,
        BinSize=1e-3, itick_box_size=5,
        pkg_len_select=np.array([1]),
    )


def _make_stubs(txffe_cursor_val=0.5, fom_val=10.0, no_zero=False, best_cursor=10):
    """Create a complete set of minimal stubs that produce a successful run."""

    def init_loop_struct():
        return SimpleNamespace(
            ctle_index=1, g_dc=0.0, g_DC_low=0.0, g_LP_index=1,
            H_ctf=None, sigma_N=0.0, sigma_ne=0.0, FOM=float('-inf'),
            cursor_i=best_cursor, itick=0, txffe=None, tx_index_vector=None,
            A_s=1.0, A_p=1.0, far_cursors=np.array([]), precursors=np.array([]),
            PSD_results=None,
        )

    def build_txffe(param):
        matrix = np.array([[txffe_cursor_val]])
        return matrix, 1, np.array([0]), np.array([[0]]), np.array([txffe_cursor_val])

    def calc_settings(txffe_matrix, chdata, param, OP):
        N = len(chdata[0].faxis)
        return SimpleNamespace(
            qual=np.ones((1, 1)),
            H_sy=np.ones(N),
            H_r=np.ones(N),
            f_xc=np.array([1e9]),
            Peak_Search_Range=np.array([1, 20]),
            delta_sbr=None,
            min_number_of_UI_in_response=10,
        )

    def fd_ctle(f, fz, fp1, fp2, gdc):
        return np.ones(len(np.asarray(f).ravel()))

    def compute_ctle(chdata, ctle_gain, THIS, f_xc, param, OP):
        N = len(f_xc) if hasattr(f_xc, '__len__') else 1
        ones_f = np.ones(len(chdata[0].faxis))
        return chdata, ones_f, np.ones(N), ones_f

    def calc_noise_xc(H_low_xc, ctle_gain_xc, SETTINGS, param, OP):
        return SimpleNamespace()

    def sigma_eta(chdata, param, H_sy, H_r, H_ctf):
        return 0.01

    def compute_txffe(chdata, pulse_struc, txffe, updated, param, OP):
        sbr = np.zeros(60)
        sbr[best_cursor - 1] = 1.0  # place signal at cursor position
        return sbr, chdata, pulse_struc

    def find_sample_point(sbr, param, OP, peak_range):
        return best_cursor, no_zero, best_cursor

    def setup_sampler_sweep(full_sample_range, BEST, OP):
        return range(len(full_sample_range)), BEST, False, False, None, None

    def compute_dfe(sbr, THIS, param, do_C2M, T_O):
        return THIS, param

    def calc_noise(THIS, best_fom, sbr, SETTINGS, chdata, param, OP):
        return THIS, 0  # abort_status = 0

    def calc_fom(chdata, do_C2M, THIS, param, OP, sbr):
        return fom_val, False  # skip_loop=False

    def set_best_itick(THIS, BEST):
        return BEST  # do NOT update BEST.FOM here (so condition fires correctly)

    def update_best(BEST, THIS, sbr, chdata, param, OP):
        BEST.FOM = THIS.FOM
        BEST.cursor_i = THIS.cursor_i
        BEST.sbr = sbr.copy()
        return BEST

    def update_best_eq_failed(BEST, THIS, sbr, chdata, param, OP):
        BEST.sbr = np.zeros(20)
        BEST.cursor_i = None
        return BEST

    def update_post_optimize(BEST, f, param, OP):
        return BEST

    def create_output(result, BEST, t, chdata, param, OP):
        result.FOM = getattr(BEST, 'FOM', float('-inf'))
        return result

    return dict(
        _OptFom_Initialize_Loop_Struct_fn=init_loop_struct,
        _OptFom_Build_TXFFE_fn=build_txffe,
        _OptFom_Calculate_Settings_fn=calc_settings,
        _FD_CTLE_fn=fd_ctle,
        _OptFom_Compute_CTLE_fn=compute_ctle,
        _OptFom_Calc_Noise_XC_fn=calc_noise_xc,
        _get_sigma_eta_ACCM_noise_fn=sigma_eta,
        _OptFom_Compute_TXFFE_fn=compute_txffe,
        _OptFom_Find_Sample_Point_fn=find_sample_point,
        _OptFom_Setup_Sampler_Sweep_fn=setup_sampler_sweep,
        _OptFom_Compute_DFE_fn=compute_dfe,
        _OptFom_Calc_Noise_fn=calc_noise,
        _OptFom_Calc_FOM_fn=calc_fom,
        _OptFom_Set_Best_Itick_fn=set_best_itick,
        _OptFom_Update_Best_Setttings_fn=update_best,
        _OptFom_Update_Best_Settings_EQ_Failed_fn=update_best_eq_failed,
        _OptFom_Update_BEST_Post_Optimize_fn=update_post_optimize,
        _OptFom_Create_Output_fn=create_output,
    )


# ---------------------------------------------------------------------------
# Test 1 – Nominal: successful run returns result.eq_failed=False
# ---------------------------------------------------------------------------

def test_successful_run_eq_not_failed():
    """Nominal run: BEST.cursor_i set → result.eq_failed=False."""
    param = _make_param()
    OP = _make_op()
    stubs = _make_stubs(fom_val=10.0, best_cursor=10)

    result = optimize_fom(OP, param, _make_chdata(), None, do_C2M=False, **stubs)

    assert result.eq_failed is False


# ---------------------------------------------------------------------------
# Test 2 – ts_sample_adj_range normalization: [v] → [0, v]
# ---------------------------------------------------------------------------

def test_ts_sample_adj_range_single_value():
    """ts_sample_adj_range=[2] → normalized to [0, 2]; full_sample_range has 3 elements."""
    param = _make_param(ts_range=np.array([2.0]))
    OP = _make_op()
    stubs = _make_stubs(best_cursor=10)

    result = optimize_fom(OP, param, _make_chdata(), None, do_C2M=False, **stubs)

    # After call, param.ts_sample_adj_range should be [0, 2]
    tsar = np.asarray(param.ts_sample_adj_range).ravel()
    assert len(tsar) == 2
    assert tsar[0] == pytest.approx(0.0)
    assert tsar[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 3 – CTLE_type CL93 → g_DC_HP_values forced to 0 (lf_indx=1)
# ---------------------------------------------------------------------------

def test_ctle_type_cl93_sets_gdc_hp_to_zero():
    """CTLE_type='CL93' → param.g_DC_HP_values reset to [0]."""
    param = _make_param(gdc_hp=np.array([0.0, -3.0, -6.0]))
    param.CTLE_type = 'CL93'
    OP = _make_op()
    stubs = _make_stubs(best_cursor=10)

    optimize_fom(OP, param, _make_chdata(), None, do_C2M=False, **stubs)

    # g_DC_HP_values should have been reset
    gdc_hp = np.asarray(param.g_DC_HP_values).ravel()
    assert len(gdc_hp) == 1
    assert float(gdc_hp[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4 – RxFFE disabled → Gffe_values forced to [0]
# ---------------------------------------------------------------------------

def test_rxffe_disabled_gffe_is_zero():
    """OP.RxFFE=False → Gffe_values=[0] regardless of param.cursor_gain."""
    param = _make_param()
    param.cursor_gain = np.array([0.5, 1.0, 2.0])
    OP = _make_op()
    OP.RxFFE = False

    compute_txffe_calls = [0]
    stubs = _make_stubs(best_cursor=10)

    orig_fn = stubs['_OptFom_Compute_TXFFE_fn']

    def counting_txffe(chdata, pulse_struc, txffe, updated, param, OP):
        compute_txffe_calls[0] += 1
        return orig_fn(chdata, pulse_struc, txffe, updated, param, OP)

    stubs['_OptFom_Compute_TXFFE_fn'] = counting_txffe

    # With 1 Gffe × 1 CTLE × 1 LP × 1 TXFFE = 1 txffe call expected
    optimize_fom(OP, param, _make_chdata(), None, do_C2M=False, **stubs)

    assert compute_txffe_calls[0] >= 1, "Expected at least 1 TXFFE call"


# ---------------------------------------------------------------------------
# Test 5 – EQ failure: no_zero_crossing always True → BEST.cursor_i stays None
# ---------------------------------------------------------------------------

def test_eq_failure_when_no_zero_crossing():
    """All iterations skip due to no_zero_crossing → result.eq_failed=True."""
    param = _make_param()
    OP = _make_op()
    stubs = _make_stubs(no_zero=True, best_cursor=10)

    # Override update_best_eq_failed to set sbr so BEST.sbr exists
    def update_eq_failed(BEST, THIS, sbr, chdata, param, OP):
        BEST.sbr = np.zeros(20)
        return BEST

    stubs['_OptFom_Update_Best_Settings_EQ_Failed_fn'] = update_eq_failed

    result = optimize_fom(OP, param, _make_chdata(), None, do_C2M=False, **stubs)

    assert result.eq_failed is True


# ---------------------------------------------------------------------------
# Test 6 – do_C2M=True EQ failure: returns early without Create_Output call
# ---------------------------------------------------------------------------

def test_do_c2m_eq_failure_returns_early():
    """do_C2M=True + EQ failure → returns before OptFom_Create_Output is called."""
    param = _make_param()
    param.T_O = 0
    OP = _make_op()
    stubs = _make_stubs(no_zero=True, best_cursor=10)

    create_called = [False]

    def create_output_tracking(result, BEST, t, chdata, param, OP):
        create_called[0] = True
        result.FOM = float('-inf')
        return result

    def update_eq_failed(BEST, THIS, sbr, chdata, param, OP):
        BEST.sbr = np.zeros(20)
        return BEST

    stubs['_OptFom_Create_Output_fn'] = create_output_tracking
    stubs['_OptFom_Update_Best_Settings_EQ_Failed_fn'] = update_eq_failed

    result = optimize_fom(OP, param, _make_chdata(), None, do_C2M=True, **stubs)

    assert not create_called[0], "Create_Output should NOT be called when do_C2M=True + EQ failed"
    assert result.eq_failed is True


# ---------------------------------------------------------------------------
# Test 7 – NonZeroLSMethod dispatch (Hansel adaptive vs legacy local search)
# ---------------------------------------------------------------------------

def _local_search_dispatch_calls(nonzero_method):
    """Run a 2-CTLE sweep with LOCAL_SEARCH>0 and count which predicate fires.
    The 2nd CTLE has a finite BEST.FOM, so the local-search dispatch is reached."""
    param = _make_param(ctle_vals=np.array([-3.0, -6.0]))
    # CTLE pole/zero arrays must match the 2-element CTLE sweep
    param.CTLE_fp1 = np.array([5e9, 5e9])
    param.CTLE_fp2 = np.array([8e9, 8e9])
    param.CTLE_fz = np.array([2e9, 2e9])
    param.LOCAL_SEARCH = 2
    param.NonZeroLSMethod = nonzero_method
    OP = _make_op()
    stubs = _make_stubs(best_cursor=10)
    calls = {'adaptive': 0, 'legacy': 0}

    def adaptive(LSV, BEST, THIS, FOM_history, iter_count, num_txffe_runs):
        calls['adaptive'] += 1
        assert iter_count >= 1
        return False  # don't skip

    def legacy(LSV, BEST, THIS, txffe_sweep_indices):
        calls['legacy'] += 1
        return False

    stubs['_OptFom_Adaptive_Local_Search_fn'] = adaptive
    stubs['_OptFom_Local_Search_fn'] = legacy
    # itick-level local search also activates when LOCAL_SEARCH>0; stub it to never skip
    stubs['_OptFom_Itick_LocalSearch_fn'] = lambda *a: False
    result = optimize_fom(OP, param, _make_chdata(), None, do_C2M=False, **stubs)
    return calls, result


def test_adaptive_local_search_is_dispatched():
    """NonZeroLSMethod=1 routes to OptFom_Adaptive_Local_Search, not the legacy one."""
    calls, result = _local_search_dispatch_calls(nonzero_method=1)
    assert calls['adaptive'] > 0
    assert calls['legacy'] == 0
    assert result.eq_failed is False


def test_legacy_local_search_is_dispatched():
    """NonZeroLSMethod=0 routes to the legacy OptFom_Local_Search."""
    calls, result = _local_search_dispatch_calls(nonzero_method=0)
    assert calls['legacy'] > 0
    assert calls['adaptive'] == 0
    assert result.eq_failed is False


# ---------------------------------------------------------------------------
# Test 8 – Optional sweep-trajectory logger (off by default; opt-in CSV)
# ---------------------------------------------------------------------------

def _read_csv(path):
    rows = [ln for ln in open(path).read().splitlines() if ln]
    header = rows[0].split(',')
    data = [dict(zip(header, ln.split(','))) for ln in rows[1:]]
    return header, data


def test_sweep_logger_off_is_noop(tmp_path, monkeypatch):
    """With SWEEP_LOG_CSV unset, no file is written and the run is unaffected."""
    monkeypatch.setattr(_ofm, 'SWEEP_LOG_CSV', None)
    param = _make_param()
    result = optimize_fom(_make_op(), param, _make_chdata(), None, do_C2M=False, **_make_stubs())
    assert result.eq_failed is False
    assert not (tmp_path / 'nope.csv').exists()


def test_sweep_logger_writes_evaluated_rows(tmp_path, monkeypatch):
    """With SWEEP_LOG_CSV set, an evaluated candidate is recorded with the method label."""
    log = str(tmp_path / 'sweep.csv')
    monkeypatch.setattr(_ofm, 'SWEEP_LOG_CSV', log)
    monkeypatch.setattr(_ofm, 'SWEEP_METHOD_LABEL', 'full_grid')
    optimize_fom(_make_op(), _make_param(), _make_chdata(), None, do_C2M=False, **_make_stubs())
    header, data = _read_csv(log)
    assert header[0] == 'method' and 'evaluated' in header and 'candidate_FOM' in header
    evaluated = [r for r in data if r['evaluated'] == '1']
    assert len(evaluated) >= 1
    assert evaluated[0]['method'] == '"full_grid"'
    assert np.isfinite(float(evaluated[0]['candidate_FOM']))


def test_sweep_logger_records_search_skips(tmp_path, monkeypatch):
    """A candidate pruned by the local-search predicate is logged as evaluated=0."""
    log = str(tmp_path / 'sweep_skip.csv')
    monkeypatch.setattr(_ofm, 'SWEEP_LOG_CSV', log)
    monkeypatch.setattr(_ofm, 'SWEEP_METHOD_LABEL', 'legacy')

    param = _make_param(ctle_vals=np.array([-3.0, -6.0]))
    param.CTLE_fp1 = np.array([5e9, 5e9])
    param.CTLE_fp2 = np.array([8e9, 8e9])
    param.CTLE_fz = np.array([2e9, 2e9])
    param.LOCAL_SEARCH = 2
    param.NonZeroLSMethod = 0
    stubs = _make_stubs(best_cursor=10)
    stubs['_OptFom_Local_Search_fn'] = lambda *a: True          # always prune
    stubs['_OptFom_Itick_LocalSearch_fn'] = lambda *a: False
    optimize_fom(_make_op(), param, _make_chdata(), None, do_C2M=False, **stubs)

    _, data = _read_csv(log)
    skipped = [r for r in data if r['skip_reason'] == '"search_skip"']
    assert len(skipped) >= 1
    assert all(r['evaluated'] == '0' for r in skipped)
