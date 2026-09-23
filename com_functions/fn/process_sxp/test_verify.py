# ============================================================
# MATLAB GROUND TRUTH
# process_sxp: orchestrates TDR computation from S-parameter chdata.
# MATLAB lines 9093–9389
#
# Key invariants:
# 1. Only chdata[0] (i=0) and package_testcase_i==1 triggers TDR computation
# 2. get_TDR is called once per (port, Z_t) combination for DD and CC modes;
#    CD and DC are only computed when Z_t == Z0
# 3. TDR11[izt] and TDR22[izt] structs are populated with ZSR, t, avgZport
# 4. ERL/ERLRMS only populated when OP.PTDR=True
# 5. With FLAG.S2P=1, only port_sel=[0] is processed → TDR11 only
# 6. OP.SHOW_BRD=True → Sfield='_raw', else Sfield='_orig'
# 7. Number of get_TDR calls:
#    S4P, Z_t=[Zt], Z0==Zt: 4 calls per port × 2 ports = 8
#    S4P, Z_t=[Zt], Z0≠Zt: 2 calls (DD+CC) per port × 2 ports = 4
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace

from com_functions.fn.process_sxp.py_impl import process_sxp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ch(N=32, fb=53.125e9):
    f = np.linspace(0.1e9, fb / 2, N)
    z = np.zeros(N, dtype=complex)
    s = 0.01 * np.ones(N, dtype=complex)
    ch = SimpleNamespace(
        faxis=f,
        sdd11_orig=s.copy(), sdd12_orig=s.copy(), sdd21_orig=s.copy(), sdd22_orig=s.copy(),
        scd11_orig=z.copy(), scd12_orig=z.copy(), scd21_orig=z.copy(), scd22_orig=z.copy(),
        sdc11_orig=z.copy(), sdc12_orig=z.copy(), sdc21_orig=z.copy(), sdc22_orig=z.copy(),
        scc11_orig=z.copy(), scc12_orig=z.copy(), scc21_orig=z.copy(), scc22_orig=z.copy(),
        sdd11_raw=s.copy(), sdd12_raw=s.copy(), sdd21_raw=s.copy(), sdd22_raw=s.copy(),
        scd11_raw=z.copy(), scd12_raw=z.copy(), scd21_raw=z.copy(), scd22_raw=z.copy(),
        sdc11_raw=z.copy(), sdc12_raw=z.copy(), sdc21_raw=z.copy(), sdc22_raw=z.copy(),
        scc11_raw=z.copy(), scc12_raw=z.copy(), scc21_raw=z.copy(), scc22_raw=z.copy(),
        base='test_ch',
    )
    return ch, f


def _make_param(Z0=50.0, Z_t=None, S2P=0, fb=53.125e9):
    if Z_t is None:
        Z_t = np.array([Z0])
    return SimpleNamespace(
        package_testcase_i=1, FLAG=SimpleNamespace(S2P=S2P),
        RL_sel=0, Z_t=Z_t, Z0=Z0,
        TR_TDR=0.025, tfx=np.array([0.0, 0.0]),
        ui=1.0 / fb, ndfe=4, N_bx=4, beta_x=0.0, Grr=1, rho_x=0.1,
        levels=4, specBER=1e-4, Tukey_Window=0,
        samples_per_ui=4, sample_dt=1.0 / (2 * 26.5625e9),
        fb=fb, fb_BT_cutoff=0.473,
    )


def _make_op():
    return SimpleNamespace(
        TDR=True, PTDR=False, DISPLAY_WINDOW=False, DEBUG=False,
        SHOW_BRD=False, AUTO_TFX=False, TDR_W_TXPKG=False,
        ERL=1, Report_Modal_ERL='disable',
        N=10, T_k=1e-9, BinSize=1e-3, cb_Guassian=True,
    )


def _stub_tdr(S, OP, param, ZT, nport):
    N_r = 8
    return SimpleNamespace(
        tdr=np.ones(N_r) * float(ZT),
        t=np.linspace(0, 1e-9, N_r),
        avgZport=float(ZT),
        ERL=10.0, ERLRMS=8.0,
        ptdr_RL=np.zeros(N_r),
        WC_ptdr_samples_t=np.zeros(1),
        WC_ptdr_samples=np.zeros(1),
    )


# ---------------------------------------------------------------------------
# Test 1 – Nominal: TDR11 and TDR22 populated for S4P
# ---------------------------------------------------------------------------

def test_tdr_fields_s4p():
    """S4P: chdata[0].TDR11[0] and TDR22[0] have ZSR, t, avgZport."""
    ch, f = _make_ch()
    param = _make_param()
    OP = _make_op()
    call_count = [0]

    def counting_tdr(S, OP_, p, ZT, np_):
        call_count[0] += 1
        return _stub_tdr(S, OP_, p, ZT, np_)

    chdata_out, _ = process_sxp(param, OP, [ch], None,
                                  _get_TDR_fn=counting_tdr)

    assert hasattr(chdata_out[0], 'TDR11')
    assert hasattr(chdata_out[0], 'TDR22')
    for field in ('ZSR', 't', 'avgZport'):
        assert hasattr(chdata_out[0].TDR11, field), f'TDR11 missing {field}'
        assert hasattr(chdata_out[0].TDR22, field), f'TDR22 missing {field}'


# ---------------------------------------------------------------------------
# Test 2 – Nominal: S2P mode only fills TDR11 (port_sel=[0])
# ---------------------------------------------------------------------------

def test_s2p_only_port1():
    """FLAG.S2P=1 → port_sel=[0] → only TDR11 is meaningfully set."""
    ch, f = _make_ch()
    param = _make_param(S2P=1)
    OP = _make_op()
    tdr_calls = []

    def recording_tdr(S, OP_, p, ZT, np_):
        tdr_calls.append(np_)
        return _stub_tdr(S, OP_, p, ZT, np_)

    chdata_out, _ = process_sxp(param, OP, [ch], None,
                                  _get_TDR_fn=recording_tdr)

    # All calls should be for port 0 (S2P mode)
    assert all(p == 0 for p in tdr_calls), f"Unexpected port calls: {tdr_calls}"
    assert hasattr(chdata_out[0], 'TDR11')


# ---------------------------------------------------------------------------
# Test 3 – Nominal: Z0==Z_t → CD and DC also computed
# ---------------------------------------------------------------------------

def test_cd_dc_computed_when_zt_equals_z0():
    """When Z_t == Z0, CD and DC TDR calls happen; otherwise only DD and CC."""
    ch, f = _make_ch()
    Z0 = 50.0
    param = _make_param(Z0=Z0, Z_t=np.array([Z0]))  # Z_t == Z0
    OP = _make_op()
    call_count = [0]

    def counting_tdr(S, OP_, p, ZT, np_):
        call_count[0] += 1
        return _stub_tdr(S, OP_, p, ZT, np_)

    process_sxp(param, OP, [ch], None, _get_TDR_fn=counting_tdr)

    # S4P, Z0==Z_t, 1 Z_t → 4 calls per port (DD+CD+DC+CC) × 2 ports = 8
    assert call_count[0] == 8, f"Expected 8 calls (got {call_count[0]})"


# ---------------------------------------------------------------------------
# Test 4 – Nominal: Z0 != Z_t → CD/DC skipped, only DD+CC
# ---------------------------------------------------------------------------

def test_cd_dc_skipped_when_zt_ne_z0():
    """When Z_t ≠ Z0, CD and DC calls are skipped (only DD and CC)."""
    ch, f = _make_ch()
    Z0 = 50.0
    Zt = 100.0  # different from Z0
    param = _make_param(Z0=Z0, Z_t=np.array([Zt]))
    OP = _make_op()
    call_count = [0]

    def counting_tdr(S, OP_, p, ZT, np_):
        call_count[0] += 1
        return _stub_tdr(S, OP_, p, ZT, np_)

    process_sxp(param, OP, [ch], None, _get_TDR_fn=counting_tdr)

    # S4P, Z0≠Z_t → 2 calls (DD+CC) per port × 2 ports = 4
    assert call_count[0] == 4, f"Expected 4 calls (got {call_count[0]})"


# ---------------------------------------------------------------------------
# Test 5 – Nominal: SHOW_BRD selects '_raw' field
# ---------------------------------------------------------------------------

def test_show_brd_uses_raw_fields():
    """OP.SHOW_BRD=True → sddXX_raw fields are used (not _orig)."""
    ch, f = _make_ch()
    # Mark _raw and _orig with different values
    ch.sdd11_raw = np.ones(len(f), dtype=complex) * 0.5
    ch.sdd11_orig = np.ones(len(f), dtype=complex) * 0.01
    param = _make_param()
    OP = _make_op()
    OP.SHOW_BRD = True

    received_s11 = {}
    call_n = [0]

    def capture_tdr(S, OP_, p, ZT, np_):
        call_n[0] += 1
        if call_n[0] == 1:  # first call = DD struct, port 0
            received_s11[0] = S.Parameters[:, 0, 0].copy()
        return _stub_tdr(S, OP_, p, ZT, np_)

    process_sxp(param, OP, [ch], None, _get_TDR_fn=capture_tdr)

    # Should have used sdd11_raw ≈ 0.5, not sdd11_orig ≈ 0.01
    assert 0 in received_s11
    np.testing.assert_allclose(
        np.abs(received_s11[0]), 0.5, atol=1e-10,
        err_msg="Expected _raw sdd11 (0.5), got _orig (0.01)"
    )


# ---------------------------------------------------------------------------
# Test 6 – Boundary: package_testcase_i != 1 → no TDR computed
# ---------------------------------------------------------------------------

def test_no_tdr_when_package_testcase_ne_1():
    """package_testcase_i=2 → skip TDR block entirely → no TDR11 attribute set."""
    ch, f = _make_ch()
    param = _make_param()
    param.package_testcase_i = 2
    OP = _make_op()
    call_count = [0]

    def counting_tdr(S, OP_, p, ZT, np_):
        call_count[0] += 1
        return _stub_tdr(S, OP_, p, ZT, np_)

    chdata_out, _ = process_sxp(param, OP, [ch], None,
                                  _get_TDR_fn=counting_tdr)

    assert call_count[0] == 0, f"Expected 0 TDR calls, got {call_count[0]}"
    assert not hasattr(chdata_out[0], 'TDR11'), "TDR11 should not be set"


# ============================================================
# COM Octave oracle values — the reference's own predicates evaluated under
# Octave on the same operands.  process_sxp itself is orchestration plus a
# large figure/uitab block, so what is pinned here is the decision logic, not
# a returned waveform.  Pinned 2026-09-22.
#
# Two divergences these pin:
#  * isequal(param.Z0, param.Z_t(izt)) is EXACT.  The port compared with a
#    1e-6 tolerance, so it ran the CD/DC TDR on impedances the reference
#    treats as unequal (and reports NaN for).
#  * find(fir4del==max(fir4del),1) for AUTO_TFX.  MATLAB's max SKIPS NaN and a
#    NaN never equals the max, so the search lands on the first non-NaN peak;
#    np.argmax returns the index of the first NaN.
#
# COM Octave:
#   isequal(100, 100)           -> 1
#   isequal(100, 100.0000005)   -> 0     (abs(a-b) < 1e-6 said True)
#   isequal(100, 99.9999995)    -> 0     (abs(a-b) < 1e-6 said True)
#   find(x==max(x),1) on [NaN 1 3 2] -> 3  (1-based; np.argmax gave 0)
#   find(x==max(x),1) on [1 3 NaN 3] -> 2  (1-based; np.argmax gave 2, the NaN)
#   find(x==max(x),1) on [NaN NaN]   -> EMPTY
# ============================================================

def test_octave_Zt_match_is_exact_not_within_1e6():
    """Z_t a hair off Z0 must be treated as DIFFERENT: no CD/DC TDR, ERL NaN."""
    ch, f = _make_ch()
    OP = _make_op()
    OP.PTDR = True
    OP.Report_Modal_ERL = 'enable'
    param = _make_param(Z0=100.0, Z_t=np.array([100.0000005]))
    zts = []

    def recording_tdr(S, OP_, p, ZT, np_):
        zts.append(ZT)
        return _stub_tdr(S, OP_, p, ZT, np_)

    out, _ = process_sxp(param, OP, [ch], None, _get_TDR_fn=recording_tdr)
    # DD at ZT and CC at ZT/4 only -- no CD (at ZT) and no DC (at ZT/4)
    assert len(zts) == 4, zts          # 2 ports x (DD + CC)
    assert np.isnan(out[0].TDR11.ERL_CD)
    assert np.isnan(out[0].TDR11.ERL_DC)


def test_octave_Zt_exactly_equal_still_runs_cd_dc():
    """Guard the other side: an exact match must still compute CD and DC."""
    ch, f = _make_ch()
    OP = _make_op()
    OP.PTDR = True
    OP.Report_Modal_ERL = 'enable'
    param = _make_param(Z0=100.0, Z_t=np.array([100.0]))
    zts = []

    def recording_tdr(S, OP_, p, ZT, np_):
        zts.append(ZT)
        return _stub_tdr(S, OP_, p, ZT, np_)

    out, _ = process_sxp(param, OP, [ch], None, _get_TDR_fn=recording_tdr)
    assert len(zts) == 8, zts          # 2 ports x (DD + CD + DC + CC)
    assert out[0].TDR11.ERL_CD == 10.0


@pytest.mark.parametrize('fir,expected_pix', [
    ([np.nan, 1.0, 3.0, 2.0], 2),
    ([1.0, 3.0, np.nan, 3.0], 1),
    ([1.0, 3.0, 2.0], 1),
])
def test_octave_auto_tfx_peak_skips_nan(fir, expected_pix):
    ch, f = _make_ch()
    OP = _make_op()
    OP.AUTO_TFX = True
    param = _make_param(Z0=100.0, Z_t=np.array([100.0]))
    tu = np.arange(len(fir), dtype=float) * 1e-12

    def fake_raw_fir(sdd21, faxis, OP_, p):
        return np.array(fir, dtype=float), tu

    out, param_out = process_sxp(param, OP, [ch], None,
                                 _get_TDR_fn=_stub_tdr,
                                 _get_RAW_FIR_fn=fake_raw_fir)
    assert param_out.tfx[1] == 2 * tu[expected_pix]


# ============================================================
# COM Octave oracle values -- the reference process_sxp itself (extracted
# verbatim from octave/com_ieee8023_4p16p0_octave_compat.m, with the real
# plot_modal alongside it and only get_TDR shimmed) run on the fixture below.
# Pinned 2026-09-23.  This closes the 'Rlcc_179mm' CM-mask copy block
# (MATLAB L9626-9636).
#
# The block itself only copies, so the numbers come from plot_modal; the test
# therefore injects the REAL plot_modal rather than a stub, which is what
# makes the pinned values the reference's and not the fixture's.
#
# One divergence this pins.  plot_modal's dB helper was
#     20*log10(abs(x) + np.finfo(float).eps)
# where the reference is
#     dB=@(x) 20*log10(squeeze(abs(x)))          (MATLAB L9407)
# with no epsilon.  The floor shifted every margin by 20/ln(10)*eps/|S| dB --
# 3.9e-14 dB at |S|=0.05, 9.6e-13 dB at |S|=0.002 -- and turned MATLAB's -Inf
# at |S|=0 into -313 dB.  Fixed in plot_modal/py_impl.py; with the epsilon
# back in, Rlcc_179mm(1) reads 24.020599913279586 against the reference's
# 24.020599913279625.
#
# The S-parameters are real-valued on purpose: Octave's abs() of a COMPLEX
# number and numpy's differ by an ULP (different hypot), which is a library
# difference and not the port's, and it would otherwise stop these arrays
# being pinned exactly.
#
# Fixture: f = [0.05 2 4 30 44 53.125 60 67 70] GHz -- one point in every
#   segment of the three masks, including both sides of 4, 44, 53.125, 60
#   and 67 GHz -- with
#   scc11 = [0.05 0.1 0.15 0.9 0.25 0.3 0.35 0.4 0.45]
#   scd22 = [0.002 0.005 0.008 0.011 0.014 0.017 0.02 0.023 0.026]
#   sdc22 = [0.004 0.007 0.01 0.013 0.016 0.019 0.022 0.025 0.028]
#   param.Z0=50, param.Z_t=100, OP.TDR=1, OP.CM_MASK_REPORT as listed.
#
# COM Octave, chdata(1) after process_sxp, CM_MASK_REPORT=1:
#   (values below, at %.17g)
#   Rlcc_179mm_fail 1   Rlcc_178mm_fail 1
#   Rlcd_179mm_fail 0   Rldc_179mm_fail 0
# COM Octave, CM_MASK_REPORT=0: plot_modal returns [] and NONE of the eight
#   fields exists on chdata(1).
# ============================================================

_OCT_F_GHZ = np.array([0.05, 2.0, 4.0, 30.0, 44.0, 53.125, 60.0, 67.0, 70.0])

_OCT_SCC11 = np.array([0.05, 0.1, 0.15, 0.9, 0.25, 0.3, 0.35, 0.4, 0.45],
                      dtype=complex)
_OCT_SCD22 = np.array([0.002, 0.005, 0.008, 0.011, 0.014, 0.017, 0.02,
                       0.023, 0.026], dtype=complex)
_OCT_SDC22 = np.array([0.004, 0.007, 0.01, 0.013, 0.016, 0.019, 0.022,
                       0.025, 0.028], dtype=complex)

_OCT_MASKS = {
    'Rlcc_179mm': [24.020599913279625, 18, 14.478174818886377,
                   -2.3848501887864977, 8.0411998265592484,
                   7.5981999056067515, 7.1186391129944884,
                   5.9588001734407516, 4.9357497244931263],
    'Rlcc_178mm': [22.770599913279625, 16.75, 13.228174818886377,
                   -2.3348501887864979, 8.7911998265592484,
                   7.2075749056067515, 5.8686391129944884,
                   4.7088001734407516, 3.6857497244931263],
    'Rlcd_179mm': [30.989753027896846, 23.434717560338449,
                   19.766435554278772, 22.383911002717852,
                   23.18802752172936, 23.391021572434525,
                   21.979400086720375, 20.765443279648146,
                   19.70053304058364],
    'Rldc_179mm': [24.969153114617221, 20.512156846773685,
                   17.828235294117647, 20.93289765974562,
                   22.028188582175623, 22.424927980943423,
                   21.151546383555875, 20.041199826559243,
                   19.056839373155615],
}
_OCT_FAIL = {'Rlcc_179mm_fail': True, 'Rlcc_178mm_fail': True,
             'Rlcd_179mm_fail': False, 'Rldc_179mm_fail': False}


def _make_ch_cm_mask():
    N = len(_OCT_F_GHZ)
    z = np.zeros(N, dtype=complex)
    s = 0.01 * np.ones(N, dtype=complex)
    return SimpleNamespace(
        faxis=_OCT_F_GHZ * 1e9,
        sdd11_orig=s.copy(), sdd12_orig=s.copy(),
        sdd21_orig=s.copy(), sdd22_orig=s.copy(),
        scd11_orig=z.copy(), scd12_orig=z.copy(),
        scd21_orig=z.copy(), scd22_orig=_OCT_SCD22.copy(),
        sdc11_orig=z.copy(), sdc12_orig=z.copy(),
        sdc21_orig=z.copy(), sdc22_orig=_OCT_SDC22.copy(),
        scc11_orig=_OCT_SCC11.copy(), scc12_orig=z.copy(),
        scc21_orig=z.copy(), scc22_orig=z.copy(),
        base='test_ch')


def _run_cm_mask(cm_mask_report):
    from com_functions.fn.plot_modal.py_impl import plot_modal
    ch = _make_ch_cm_mask()
    param = _make_param(Z0=50.0, Z_t=np.array([100.0]))
    OP = _make_op()
    OP.CM_MASK_REPORT = cm_mask_report
    OP.PLOT_CM = False
    out, _ = process_sxp(param, OP, [ch], None,
                         _plot_modal_fn=plot_modal, _get_TDR_fn=_stub_tdr)
    return out


@pytest.mark.parametrize('field', sorted(_OCT_MASKS))
def test_octave_cm_mask_margins_copied_to_chdata(field):
    """plot_modal's margin arrays land on chdata[0] bit for bit.

    Exercises the `if ~isempty(return_struct)` copy block, and pins the
    numbers it carries to the reference run under Octave.
    """
    out = _run_cm_mask(True)
    got = np.asarray(getattr(out[0], field), dtype=float).ravel()
    np.testing.assert_array_equal(got, np.asarray(_OCT_MASKS[field]))


def test_octave_cm_mask_fail_flags_copied_to_chdata():
    """The four pass/fail booleans are copied with the arrays."""
    out = _run_cm_mask(True)
    for field, expected in _OCT_FAIL.items():
        assert bool(getattr(out[0], field)) is expected, field


def test_octave_cm_mask_fields_absent_when_report_off():
    """CM_MASK_REPORT=0: plot_modal returns [], so the copy block is skipped
    and chdata(1) gains none of the eight fields."""
    out = _run_cm_mask(False)
    for field in list(_OCT_MASKS) + list(_OCT_FAIL):
        assert not hasattr(out[0], field), field


def test_octave_auto_tfx_all_nan_raises():
    """COM Octave: find(x==max(x),1) on an all-NaN vector is EMPTY, so
    param.tfx(2)=2*tu([]) is not a value assignment at all."""
    ch, f = _make_ch()
    OP = _make_op()
    OP.AUTO_TFX = True
    param = _make_param(Z0=100.0, Z_t=np.array([100.0]))

    def fake_raw_fir(sdd21, faxis, OP_, p):
        return np.full(4, np.nan), np.arange(4, dtype=float) * 1e-12

    with pytest.raises(ValueError, match='all-NaN'):
        process_sxp(param, OP, [ch], None, _get_TDR_fn=_stub_tdr,
                    _get_RAW_FIR_fn=fake_raw_fir)
