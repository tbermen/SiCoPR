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
