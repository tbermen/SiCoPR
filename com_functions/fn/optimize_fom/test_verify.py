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

import contextlib
import io

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

    def adaptive(LSV, BEST, THIS, FOM_history, iter_count, num_txffe_runs,
                 Overwrite_Min_Radius=None, matlab_version='4p15p0'):
        calls['adaptive'] += 1
        assert iter_count >= 1
        # 4p16p0 added these two; optimize_fom must forward them so the mainline
        # min_radius rule and the config override can take effect.
        calls['saw_version'] = matlab_version
        calls['saw_min_radius'] = Overwrite_Min_Radius
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


# --------------------------------------------------------------------------
# The orchestration itself, against COM Octave.
#
# Every test above drives optimize_fom through stubs and asserts an invariant
# read off the MATLAB. This block asserts the same thing the harder way: it
# runs the REFERENCE optimize_fom under Octave through an arithmetically
# identical stub set and pins the resulting CALL TRACE, row for row.
#
# That is the right instrument for this function. optimize_fom is almost
# entirely control flow -- five nested loops, nine skip conditions, one break,
# and 20 helper calls. All of its arithmetic lives in those helpers, and every
# one of them is separately oracle-backed. What is left to get wrong is the
# ORDER and the SKIPS, and a trace that matches row for row is a direct
# statement that the port takes the same path through the search.
#
# Each row is (helper, ctle_index, g_LP_index, itick, cursor_i, txffe c(0), FOM)
# with NA where a field is not meaningful at that call site. Three things in
# the trace come from real reference logic that no stub can fake:
#
#   the sigma_ISI_ignoreDFE prune (ML 8686) fires off the stub pulse and, in
#   the symmetric scenario, drops every itick > 0 once a best FOM exists --
#   visible as a Compute_TXFFE with only one Compute_DFE after it. The causal
#   scenario uses a pulse that is zero before the cursor, which zeroes the
#   precursors and disables the prune, so the rest of the loop body is reached;
#
#   abort_status 1 continues the itick loop and 2 breaks it. In the causal
#   scenario c(0)=0.9 aborts with 1 at itick=1 and itick=2 still runs, while
#   c(0)=0.8 aborts with 2 at itick=1 and itick=2 never appears. A break
#   rewritten as a continue changes the trace and is caught;
#
#   Set_Best_Itick is called for every candidate but Update_Best_Setttings
#   only for a winner, so a losing candidate shows a row 14 with no row 15.
#
# cursor_i is the one place the two differ on purpose: the reference is
# 1-based, the port 0-based (AUDIT FINDING B16-D20), so the port value is
# compared as cursor_i + 1 rather than being hidden in a tolerance.
#
# Generated 2026-09-23 against COM Octave 4p16p0. Re-generate by re-running
# the reference through matching .m stubs, never by copying what the port
# prints.
# --------------------------------------------------------------------------

NA = None           # field not meaningful at this call site

_HELPER = {0: 'Initialize_Loop_Struct', 1: 'Build_TXFFE', 2: 'Calculate_Settings',
           3: 'FD_CTLE', 4: 'Compute_CTLE', 5: 'Calc_Noise_XC',
           6: 'get_sigma_eta_ACCM_noise', 7: 'Compute_TXFFE',
           8: 'Find_Sample_Point', 9: 'Setup_Sampler_Sweep',
           10: 'Itick_LocalSearch', 11: 'Compute_DFE', 12: 'Calc_Noise',
           13: 'Calc_FOM', 14: 'Set_Best_Itick', 15: 'Update_Best_Setttings',
           16: 'Update_Best_Settings_EQ_Failed', 17: 'Update_BEST_Post_Optimize',
           18: 'Create_Output', 19: 'Local_Search'}


_TRACE_SYMMETRIC_PRUNE = (
    ( 0, NA, NA, NA, NA, NA, NA),
    ( 1, NA, NA, NA, NA, NA, NA),
    ( 2, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.9, NA),
    (12, 1, 1, 0, 40, 0.9, NA),
    (13, 1, 1, 0, 40, 0.9, 10.15),
    (14, 1, 1, 0, 40, 0.9, 10.15),
    (15, 1, 1, 0, 40, 0.9, 10.15),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 4, 1, 2, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 2, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (10, NA, NA, 1, NA, NA, NA),
    (10, NA, NA, 2, NA, NA, NA),
    (17, NA, NA, NA, NA, NA, 1000),
    (18, NA, NA, NA, NA, NA, 120),
)

_TRACE_CAUSAL_NO_PRUNE = (
    ( 0, NA, NA, NA, NA, NA, NA),
    ( 1, NA, NA, NA, NA, NA, NA),
    ( 2, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.9, NA),
    (12, 1, 1, 0, 40, 0.9, NA),
    (13, 1, 1, 0, 40, 0.9, 10.15),
    (14, 1, 1, 0, 40, 0.9, 10.15),
    (15, 1, 1, 0, 40, 0.9, 10.15),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 1, 1, 41, 0.9, NA),
    (12, 1, 1, 1, 41, 0.9, NA),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 1, 2, 42, 0.9, NA),
    (12, 1, 1, 2, 42, 0.9, NA),
    (13, 1, 1, 2, 42, 0.9, 10.15),
    (14, 1, 1, 2, 42, 0.9, 10.15),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.8, NA),
    (12, 1, 1, 0, 40, 0.8, NA),
    (13, 1, 1, 0, 40, 0.8, 10.05),
    (14, 1, 1, 0, 40, 0.8, 10.05),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 1, 1, 41, 0.8, NA),
    (12, 1, 1, 1, 41, 0.8, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.7, NA),
    (12, 1, 1, 0, 40, 0.7, NA),
    (13, 1, 1, 0, 40, 0.7, 9.95),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 1, 1, 41, 0.7, NA),
    (12, 1, 1, 1, 41, 0.7, NA),
    (13, 1, 1, 1, 41, 0.7, 10.45),
    (14, 1, 1, 1, 41, 0.7, 10.45),
    (15, 1, 1, 1, 41, 0.7, 10.45),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 1, 2, 42, 0.7, NA),
    (12, 1, 1, 2, 42, 0.7, NA),
    (13, 1, 1, 2, 42, 0.7, 9.95),
    (14, 1, 1, 2, 42, 0.7, 9.95),
    ( 4, 1, 2, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 2, 0, 40, 0.9, NA),
    (12, 1, 2, 0, 40, 0.9, NA),
    (13, 1, 2, 0, 40, 0.9, 10.15),
    (14, 1, 2, 0, 40, 0.9, 10.15),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 2, 1, 41, 0.9, NA),
    (12, 1, 2, 1, 41, 0.9, NA),
    (13, 1, 2, 1, 41, 0.9, 10.65),
    (14, 1, 2, 1, 41, 0.9, 10.65),
    (15, 1, 2, 1, 41, 0.9, 10.65),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 2, 2, 42, 0.9, NA),
    (12, 1, 2, 2, 42, 0.9, NA),
    (13, 1, 2, 2, 42, 0.9, 10.15),
    (14, 1, 2, 2, 42, 0.9, 10.15),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 2, 0, 40, 0.8, NA),
    (12, 1, 2, 0, 40, 0.8, NA),
    (13, 1, 2, 0, 40, 0.8, 10.05),
    (14, 1, 2, 0, 40, 0.8, 10.05),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 2, 1, 41, 0.8, NA),
    (12, 1, 2, 1, 41, 0.8, NA),
    (13, 1, 2, 1, 41, 0.8, 10.55),
    (14, 1, 2, 1, 41, 0.8, 10.55),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 2, 2, 42, 0.8, NA),
    (12, 1, 2, 2, 42, 0.8, NA),
    (13, 1, 2, 2, 42, 0.8, 10.05),
    (14, 1, 2, 2, 42, 0.8, 10.05),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 2, 0, 40, 0.7, NA),
    (12, 1, 2, 0, 40, 0.7, NA),
    (13, 1, 2, 0, 40, 0.7, 9.95),
    (14, 1, 2, 0, 40, 0.7, 9.95),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 2, 1, 41, 0.7, NA),
    (12, 1, 2, 1, 41, 0.7, NA),
    (13, 1, 2, 1, 41, 0.7, 10.45),
    (14, 1, 2, 1, 41, 0.7, 10.45),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 2, 2, 42, 0.7, NA),
    (12, 1, 2, 2, 42, 0.7, NA),
    (13, 1, 2, 2, 42, 0.7, 9.95),
    (14, 1, 2, 2, 42, 0.7, 9.95),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 2, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 2, 1, 0, 40, 0.9, NA),
    (12, 2, 1, 0, 40, 0.9, NA),
    (13, 2, 1, 0, 40, 0.9, 4.15),
    (14, 2, 1, 0, 40, 0.9, 4.15),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 2, 1, 1, 41, 0.9, NA),
    (12, 2, 1, 1, 41, 0.9, NA),
    (13, 2, 1, 1, 41, 0.9, 4.65),
    (14, 2, 1, 1, 41, 0.9, 4.65),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 2, 1, 2, 42, 0.9, NA),
    (12, 2, 1, 2, 42, 0.9, NA),
    (13, 2, 1, 2, 42, 0.9, 4.15),
    (14, 2, 1, 2, 42, 0.9, 4.15),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 2, 1, 0, 40, 0.8, NA),
    (12, 2, 1, 0, 40, 0.8, NA),
    (13, 2, 1, 0, 40, 0.8, 4.05),
    (14, 2, 1, 0, 40, 0.8, 4.05),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 2, 1, 1, 41, 0.8, NA),
    (12, 2, 1, 1, 41, 0.8, NA),
    (13, 2, 1, 1, 41, 0.8, 4.55),
    (14, 2, 1, 1, 41, 0.8, 4.55),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 2, 1, 2, 42, 0.8, NA),
    (12, 2, 1, 2, 42, 0.8, NA),
    (13, 2, 1, 2, 42, 0.8, 4.05),
    (14, 2, 1, 2, 42, 0.8, 4.05),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 2, 1, 0, 40, 0.7, NA),
    (12, 2, 1, 0, 40, 0.7, NA),
    (13, 2, 1, 0, 40, 0.7, 3.95),
    (14, 2, 1, 0, 40, 0.7, 3.95),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 2, 1, 1, 41, 0.7, NA),
    (12, 2, 1, 1, 41, 0.7, NA),
    (13, 2, 1, 1, 41, 0.7, 4.45),
    (14, 2, 1, 1, 41, 0.7, 4.45),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 2, 1, 2, 42, 0.7, NA),
    (12, 2, 1, 2, 42, 0.7, NA),
    (13, 2, 1, 2, 42, 0.7, 3.95),
    (14, 2, 1, 2, 42, 0.7, 3.95),
    (17, NA, NA, NA, NA, NA, 1000),
    (18, NA, NA, NA, NA, NA, 120),
)

_TRACE_LOCAL_SEARCH_2 = (
    ( 0, NA, NA, NA, NA, NA, NA),
    ( 1, NA, NA, NA, NA, NA, NA),
    ( 2, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.9, NA),
    (12, 1, 1, 0, 40, 0.9, NA),
    (13, 1, 1, 0, 40, 0.9, 10.15),
    (14, 1, 1, 0, 40, 0.9, 10.15),
    (15, 1, 1, 0, 40, 0.9, 10.15),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 1, 1, 41, 0.9, NA),
    (12, 1, 1, 1, 41, 0.9, NA),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 1, 2, 42, 0.9, NA),
    (12, 1, 1, 2, 42, 0.9, NA),
    (13, 1, 1, 2, 42, 0.9, 10.15),
    (14, 1, 1, 2, 42, 0.9, 10.15),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.8, NA),
    (12, 1, 1, 0, 40, 0.8, NA),
    (13, 1, 1, 0, 40, 0.8, 10.05),
    (14, 1, 1, 0, 40, 0.8, 10.05),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 1, 1, 41, 0.8, NA),
    (12, 1, 1, 1, 41, 0.8, NA),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 1, 0, 40, 0.7, NA),
    (12, 1, 1, 0, 40, 0.7, NA),
    (13, 1, 1, 0, 40, 0.7, 9.95),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 1, 1, 41, 0.7, NA),
    (12, 1, 1, 1, 41, 0.7, NA),
    (13, 1, 1, 1, 41, 0.7, 10.45),
    (14, 1, 1, 1, 41, 0.7, 10.45),
    (15, 1, 1, 1, 41, 0.7, 10.45),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 1, 2, 42, 0.7, NA),
    (12, 1, 1, 2, 42, 0.7, NA),
    (13, 1, 1, 2, 42, 0.7, 9.95),
    (14, 1, 1, 2, 42, 0.7, 9.95),
    ( 4, 1, 2, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 2, 0, 40, 0.9, NA),
    (12, 1, 2, 0, 40, 0.9, NA),
    (13, 1, 2, 0, 40, 0.9, 10.15),
    (14, 1, 2, 0, 40, 0.9, 10.15),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 2, 1, 41, 0.9, NA),
    (12, 1, 2, 1, 41, 0.9, NA),
    (13, 1, 2, 1, 41, 0.9, 10.65),
    (14, 1, 2, 1, 41, 0.9, 10.65),
    (15, 1, 2, 1, 41, 0.9, 10.65),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 2, 2, 42, 0.9, NA),
    (12, 1, 2, 2, 42, 0.9, NA),
    (13, 1, 2, 2, 42, 0.9, 10.15),
    (14, 1, 2, 2, 42, 0.9, 10.15),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 2, 0, 40, 0.8, NA),
    (12, 1, 2, 0, 40, 0.8, NA),
    (13, 1, 2, 0, 40, 0.8, 10.05),
    (14, 1, 2, 0, 40, 0.8, 10.05),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 2, 1, 41, 0.8, NA),
    (12, 1, 2, 1, 41, 0.8, NA),
    (13, 1, 2, 1, 41, 0.8, 10.55),
    (14, 1, 2, 1, 41, 0.8, 10.55),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 2, 2, 42, 0.8, NA),
    (12, 1, 2, 2, 42, 0.8, NA),
    (13, 1, 2, 2, 42, 0.8, 10.05),
    (14, 1, 2, 2, 42, 0.8, 10.05),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 1, 2, 0, 40, 0.7, NA),
    (12, 1, 2, 0, 40, 0.7, NA),
    (13, 1, 2, 0, 40, 0.7, 9.95),
    (14, 1, 2, 0, 40, 0.7, 9.95),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 1, 2, 1, 41, 0.7, NA),
    (12, 1, 2, 1, 41, 0.7, NA),
    (13, 1, 2, 1, 41, 0.7, 10.45),
    (14, 1, 2, 1, 41, 0.7, 10.45),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 1, 2, 2, 42, 0.7, NA),
    (12, 1, 2, 2, 42, 0.7, NA),
    (13, 1, 2, 2, 42, 0.7, 9.95),
    (14, 1, 2, 2, 42, 0.7, 9.95),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 2, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.9, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 2, 1, 0, 40, 0.9, NA),
    (12, 2, 1, 0, 40, 0.9, NA),
    (13, 2, 1, 0, 40, 0.9, 4.15),
    (14, 2, 1, 0, 40, 0.9, 4.15),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 2, 1, 1, 41, 0.9, NA),
    (12, 2, 1, 1, 41, 0.9, NA),
    (13, 2, 1, 1, 41, 0.9, 4.65),
    (14, 2, 1, 1, 41, 0.9, 4.65),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 2, 1, 2, 42, 0.9, NA),
    (12, 2, 1, 2, 42, 0.9, NA),
    (13, 2, 1, 2, 42, 0.9, 4.15),
    (14, 2, 1, 2, 42, 0.9, 4.15),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.8, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 2, 1, 0, 40, 0.8, NA),
    (12, 2, 1, 0, 40, 0.8, NA),
    (13, 2, 1, 0, 40, 0.8, 4.05),
    (14, 2, 1, 0, 40, 0.8, 4.05),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 2, 1, 1, 41, 0.8, NA),
    (12, 2, 1, 1, 41, 0.8, NA),
    (13, 2, 1, 1, 41, 0.8, 4.55),
    (14, 2, 1, 1, 41, 0.8, 4.55),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 2, 1, 2, 42, 0.8, NA),
    (12, 2, 1, 2, 42, 0.8, NA),
    (13, 2, 1, 2, 42, 0.8, 4.05),
    (14, 2, 1, 2, 42, 0.8, 4.05),
    (19, NA, NA, NA, NA, NA, NA),
    ( 7, NA, NA, NA, NA, 0.7, NA),
    ( 8, NA, NA, NA, NA, NA, NA),
    ( 9, NA, NA, NA, NA, NA, NA),
    (10, NA, NA, 0, NA, NA, NA),
    (11, 2, 1, 0, 40, 0.7, NA),
    (12, 2, 1, 0, 40, 0.7, NA),
    (13, 2, 1, 0, 40, 0.7, 3.95),
    (14, 2, 1, 0, 40, 0.7, 3.95),
    (10, NA, NA, 1, NA, NA, NA),
    (11, 2, 1, 1, 41, 0.7, NA),
    (12, 2, 1, 1, 41, 0.7, NA),
    (13, 2, 1, 1, 41, 0.7, 4.45),
    (14, 2, 1, 1, 41, 0.7, 4.45),
    (10, NA, NA, 2, NA, NA, NA),
    (11, 2, 1, 2, 42, 0.7, NA),
    (12, 2, 1, 2, 42, 0.7, NA),
    (13, 2, 1, 2, 42, 0.7, 3.95),
    (14, 2, 1, 2, 42, 0.7, 3.95),
    (17, NA, NA, NA, NA, NA, 1000),
    (18, NA, NA, NA, NA, NA, 120),
)

_TRACE_EQ_FAILED = (
    ( 0, NA, NA, NA, NA, NA, NA),
    ( 1, NA, NA, NA, NA, NA, NA),
    ( 2, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 2, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 2, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    (16, NA, NA, NA, NA, NA, NA),
    (17, NA, NA, NA, NA, NA, 1000),
    (18, NA, NA, NA, NA, NA, 20),
)

_TRACE_EQ_FAILED_C2M = (
    ( 0, NA, NA, NA, NA, NA, NA),
    ( 1, NA, NA, NA, NA, NA, NA),
    ( 2, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 2, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 2, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 4, 1, 2, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 3, NA, NA, NA, NA, NA, NA),
    ( 4, 2, 1, NA, NA, NA, NA),
    ( 5, NA, NA, NA, NA, NA, NA),
    ( 6, NA, NA, NA, NA, NA, NA),
    (16, NA, NA, NA, NA, NA, NA),
)


_OCT_NCH = 32
_OCT_FB = 53.125e9
_OCT_M = 4

# name, LOCAL_SEARCH, tx_ffe_c0_min, do_C2M, pulse width, pulse zeroed below,
# reference trace, eq_failed, result.FOM (None = no Create_Output), len(t)
_OCT_SCEN = (
    ('symmetric pulse, the ISI prune dominates',
     0, 0.6, False, 8.0, 0.0, _TRACE_SYMMETRIC_PRUNE, False, 10.15, 120),
    ('causal pulse, no prune: abort 1, abort 2 and a losing FOM all reached',
     0, 0.6, False, 6.0, 39.0, _TRACE_CAUSAL_NO_PRUNE, False, 10.65, 120),
    ('LOCAL_SEARCH=2',
     2, 0.6, False, 6.0, 39.0, _TRACE_LOCAL_SEARCH_2, False, 10.65, 120),
    ('every TXFFE below tx_ffe_c0_min: eq_failed',
     0, 2.0, False, 8.0, 0.0, _TRACE_EQ_FAILED, True, float('-inf'), 20),
    ('eq_failed with do_C2M: early return',
     0, 2.0, True, 8.0, 0.0, _TRACE_EQ_FAILED_C2M, True, None, 0),
)


def _oct_sbr(txffe, width, zero_below):
    """The stub pulse: sbr(k), k = 1..120, identical in both languages.

    zero_below kills the samples ahead of the cursor, which is what disables
    the sigma_ISI prune: with no precursors the ISI norm is ~1e-5 of A_s.
    """
    k = np.arange(1, 121, dtype=float)
    v = np.exp(-((k - 40) / width) ** 2) * (1.0 + 0.05 * float(txffe[1]))
    v[k < zero_below] = 0.0
    return v


def _oct_stubs(log, width, zero_below):
    """Python twins of the Octave stub .m files the reference was run against.

    Each is deterministic and side-effect free apart from the trace row it
    appends, so a trace difference is a difference in optimize_fom and nowhere
    else.
    """
    def rec(code, THIS, fom=None):
        tx = THIS.txffe
        log.append((code, float(THIS.ctle_index), float(THIS.g_LP_index),
                    float(THIS.itick),
                    # 0-BASED here, 1-based in the reference: see B16-D20
                    float(THIS.cursor_i) + 1.0,
                    float(np.ravel(tx)[1]) if tx is not None else None,
                    None if fom is None else float(fom)))

    def flat(code, *vals):
        log.append((code,) + tuple(vals) + (None,) * (6 - len(vals)))

    def init_loop_struct():
        flat(0)
        return SimpleNamespace(
            ctle_index=1, g_dc=0.0, g_DC_low=0.0, g_LP_index=1, H_ctf=None,
            sigma_N=0.0, sigma_ne=0.0, FOM=float('-inf'), cursor_i=None,
            itick=0, txffe=None, tx_index_vector=None, A_s=1.0, A_p=1.0,
            far_cursors=np.array([]), precursors=np.array([]), PSD_results=None)

    def build_txffe(param):
        flat(1)
        return (np.array([[-0.05, 0.90, -0.05], [-0.20, 0.50, -0.30],
                          [-0.10, 0.80, -0.10], [-0.15, 0.70, -0.15]]),
                2, np.array([1, 3]),
                np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]),
                np.array([0.90, 0.50, 0.80, 0.70]))

    def calc_settings(txffe_matrix, chdata, param, OP):
        flat(2)
        n = len(chdata[0].faxis)
        return SimpleNamespace(
            qual=np.array([[1.0, 1.0], [1.0, 0.0]]),   # (2,2) disqualified
            H_sy=np.ones(n), H_r=np.ones(n), f_xc=np.array([1e9]),
            Peak_Search_Range=np.array([1, 60]), delta_sbr=None,
            min_number_of_UI_in_response=10)

    def fd_ctle(f, fz, fp1, fp2, gdc):
        flat(3)
        return np.ones(len(np.ravel(np.asarray(f))))

    def compute_ctle(chdata, ctle_gain, THIS, f_xc, param, OP):
        flat(4, float(THIS.ctle_index), float(THIS.g_LP_index))
        n = len(chdata[0].faxis)
        return chdata, np.ones(n), np.ones(len(np.ravel(f_xc))), np.ones(n)

    def calc_noise_xc(H_low_xc, ctle_gain_xc, SETTINGS, param, OP):
        flat(5)
        return SimpleNamespace(placeholder=0)

    def sigma_eta(chdata, param, H_sy, H_r, H_ctf):
        flat(6)
        return 0.01

    def compute_txffe(chdata, pulse_struc, txffe, updated, param, OP):
        flat(7, None, None, None, None, float(np.ravel(txffe)[1]))
        return _oct_sbr(np.ravel(txffe), width, zero_below), chdata, pulse_struc

    def find_sample_point(sbr, param, OP, peak_range):
        flat(8)
        return 39, False, 39          # 0-BASED; the Octave stub returns 40

    def setup_sampler_sweep(full_sample_range, BEST, OP):
        flat(9)
        return range(len(full_sample_range)), BEST, False, False, None, None

    def itick_local_search(itick, middle_search, BEST, ls_value):
        flat(10, None, None, float(itick))
        return 0

    def local_search(ls_value, BEST, THIS, sweep_indices):
        flat(19)
        return 0

    def compute_dfe(sbr, THIS, param, do_C2M, T_O):
        rec(11, THIS)
        return THIS, param

    def calc_noise(THIS, best_fom, sbr, SETTINGS, chdata, param, OP):
        rec(12, THIS)
        abort = 0
        first = THIS.ctle_index == 1 and THIS.g_LP_index == 1
        # abort 1 continues the itick loop, abort 2 breaks it: both reachable
        if first and abs(float(THIS.txffe[1]) - 0.9) < 1e-12 and THIS.itick == 1:
            abort = 1
        if first and abs(float(THIS.txffe[1]) - 0.8) < 1e-12 and THIS.itick == 1:
            abort = 2
        return THIS, abort

    def calc_fom(chdata, do_C2M, THIS, param, OP, sbr):
        fom = (10 - (THIS.g_dc + 3.5) ** 2 - 0.5 * (THIS.itick - 1) ** 2
               + float(THIS.txffe[1]))
        skip = bool(THIS.ctle_index == 1 and THIS.g_LP_index == 1
                    and abs(float(THIS.txffe[1]) - 0.7) < 1e-12
                    and THIS.itick == 0)
        rec(13, THIS, fom)
        return fom, skip

    def set_best_itick(THIS, BEST):
        rec(14, THIS, THIS.FOM)
        return BEST

    def update_best(BEST, THIS, sbr, chdata, param, OP):
        rec(15, THIS, THIS.FOM)
        BEST.FOM = THIS.FOM
        BEST.cursor_i = THIS.cursor_i
        BEST.sbr = np.asarray(sbr).copy()
        return BEST

    def update_best_eq_failed(BEST, THIS, sbr, chdata, param, OP):
        flat(16)
        BEST.sbr = np.zeros(20)
        BEST.cursor_i = None
        return BEST

    def update_post_optimize(BEST, f, param, OP):
        flat(17, None, None, None, None, None, float(len(np.ravel(f))))
        return BEST

    def create_output(result, BEST, t, chdata, param, OP):
        flat(18, None, None, None, None, None, float(len(np.ravel(t))))
        result.FOM = getattr(BEST, 'FOM', float('-inf'))
        result.t_axis = np.asarray(t)
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
        _OptFom_Itick_LocalSearch_fn=itick_local_search,
        _OptFom_Local_Search_fn=local_search,
        _OptFom_Compute_DFE_fn=compute_dfe,
        _OptFom_Calc_Noise_fn=calc_noise,
        _OptFom_Calc_FOM_fn=calc_fom,
        _OptFom_Set_Best_Itick_fn=set_best_itick,
        _OptFom_Update_Best_Setttings_fn=update_best,
        _OptFom_Update_Best_Settings_EQ_Failed_fn=update_best_eq_failed,
        _OptFom_Update_BEST_Post_Optimize_fn=update_post_optimize,
        _OptFom_Create_Output_fn=create_output)


def _oct_run(local_search_val, c0_min, do_C2M, width, zero_below):
    param = SimpleNamespace(
        ctle_gdc_values=np.array([-3.0, -6.0]),
        g_DC_HP_values=np.array([0.0, 1.0]),
        CTLE_fp1=np.array([5e9, 5e9]), CTLE_fp2=np.array([8e9, 8e9]),
        CTLE_fz=np.array([2e9, 2e9]), cursor_gain=np.array([1.0]),
        ts_sample_adj_range=np.array([2.0]),
        ndfe=4, N_bmax=6, Floating_DFE=False,
        samples_per_ui=_OCT_M, ui=1.0 / _OCT_FB, sigma_X=1.0, R_LM=1.0,
        levels=4, GDC_MIN=0, LOCAL_SEARCH=local_search_val, NonZeroLSMethod=0,
        tx_ffe_c0_min=c0_min, T_O=0, fb=_OCT_FB, CTLE_type='generic',
        cursor_index=1, Min_VEO_Test=1, matlab_version='4p16p0')
    OP = SimpleNamespace(
        RxFFE=False, FFE_OPT_METHOD='sweep', RxFFE_with_MMSE=0,
        INCLUDE_CTLE=1, RX_CALIBRATION=False, DISPLAY_WINDOW=False,
        Optimize_loop_speed_up=0, BinSize=1e-3, itick_box_size=5)
    chdata = [SimpleNamespace(
        faxis=np.linspace(0, _OCT_FB / 2, _OCT_NCH),
        sdd21f=np.ones(_OCT_NCH, dtype=complex) * 0.5, type='THRU')]
    log = []
    with contextlib.redirect_stdout(io.StringIO()):
        result = optimize_fom(OP, param, chdata, None, do_C2M,
                              **_oct_stubs(log, width, zero_below))
    return result, log, param, OP


def _fmt(row):
    return '%-26s %s' % (_HELPER.get(int(row[0]), '?'),
                         ' '.join('.' if v is None else '%g' % v
                                  for v in row[1:]))


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == b or abs(a - b) <= 1e-9 * max(abs(a), 1.0)


@pytest.mark.parametrize(
    'name,ls,c0,c2m,width,zlo,want,_ef,_fom,_nt', _OCT_SCEN,
    ids=[s[0] for s in _OCT_SCEN])
def test_octave_call_trace(name, ls, c0, c2m, width, zlo, want, _ef, _fom, _nt):
    """Every helper call, in order, with its loop state -- against COM Octave.

    The reference calls OptFom_Itick_LocalSearch unconditionally while the port
    guards it on LOCAL_SEARCH > 0, so rows for that helper are dropped from
    BOTH sides when LOCAL_SEARCH == 0. That is the one deviation and it cannot
    change a result: the reference function's own body leaves skip_it at 0
    whenever LocalSearch_Value == 0. The LOCAL_SEARCH=2 scenario keeps those
    rows and compares them for real.
    """
    _, got, _, _ = _oct_run(ls, c0, c2m, width, zlo)
    if ls == 0:
        want = tuple(r for r in want if r[0] != 10)
        got = [r for r in got if r[0] != 10]

    for i in range(min(len(want), len(got))):
        if not all(_same(a, b) for a, b in zip(want[i], got[i])):
            raise AssertionError(
                'call %d differs\n  COM Octave %s\n  port       %s'
                % (i, _fmt(want[i]), _fmt(got[i])))
    assert len(got) == len(want), (
        'port made %d helper calls, COM Octave made %d; first unmatched is %s'
        % (len(got), len(want),
           _fmt((got if len(got) > len(want) else want)[min(len(want),
                                                            len(got))])))


@pytest.mark.parametrize(
    'name,ls,c0,c2m,width,zlo,_w,want_ef,want_fom,want_nt', _OCT_SCEN,
    ids=[s[0] for s in _OCT_SCEN])
def test_octave_outputs(name, ls, c0, c2m, width, zlo, _w,
                        want_ef, want_fom, want_nt):
    """eq_failed, the FOM that survives the search, and the time axis."""
    result, _, param, _ = _oct_run(ls, c0, c2m, width, zlo)
    assert result.eq_failed is want_ef

    if want_fom is None:
        # do_C2M with eq_failed returns BEFORE OptFom_Create_Output (ML 8863),
        # so the result carries eq_failed and nothing else
        assert not hasattr(result, 'FOM'), (
            'the port ran past the do_C2M early return: result has FOM=%r'
            % result.FOM)
    else:
        assert _same(float(result.FOM), float(want_fom)), (
            'FOM %.17g, COM Octave gives %.17g' % (result.FOM, want_fom))

    t = np.ravel(np.asarray(getattr(result, 't_axis', np.array([]))))
    assert t.size == want_nt, ('t has %d samples, COM Octave gives %d'
                               % (t.size, want_nt))
    if t.size:
        # ML 8870: t = 0:ui/M:(n-1)*ui/M, compared bit-exactly rather than
        # closely, because a colon expression and an arange can disagree both
        # in the last bits and in the element COUNT.
        step = (1.0 / _OCT_FB) / _OCT_M
        assert t[-1] == (t.size - 1) * step
        assert float(np.max(np.abs(t - np.arange(t.size) * step))) == 0.0

    # param is by value in MATLAB and by reference here, so these are the
    # mutations optimize_fom makes to its own copy and reads back later
    assert list(np.ravel(param.ts_sample_adj_range)) == [0.0, 2.0]
    assert int(param.ndfe) == 4 and int(param.ndfe_passed) == 4
