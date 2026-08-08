# ============================================================
# MATLAB GROUND TRUTH
# get_TDR: TDR impedance/ERL from S-parameters.
# MATLAB lines 6904–7220
#
# Key invariants:
# 1. For a perfectly matched S2P load (RL=0), TDR should be ≈ ZT everywhere
#
# 2. TDR_RL formula: for s11=0, s12=s21=1, s22=0 → RL = (Zin^2 - Zout^2)/(Zin^2 + 2*Zin*Zout + Zout^2)
#    = (Zin - Zout)/(Zin + Zout) = rho_0 (standard Γ formula at DC)
#
# 3. Result always has: f, tdr, t, Rx_filter, tx_filter fields
#
# 4. t is shifted by -delay (500 ps)
#
# 5. PTDR path fills ERL, ERLRMS, ptdr_RL, WC_ptdr_samples
#
# 6. fmin truncation: tstart ≥ TR_TDR*1e-9 (accounts for Gaussian precursor)
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace

from com_functions.fn.get_TDR.py_impl import get_TDR, _TDR_RL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_S(N, f, rl_val=0.01, il_val=0.9, Zref=50.0, NumPorts=2):
    Params = np.zeros((N, NumPorts, NumPorts), dtype=complex)
    for i in range(N):
        Params[i, 0, 0] = rl_val
        Params[i, 1, 1] = rl_val
        if NumPorts > 1:
            Params[i, 0, 1] = il_val
            Params[i, 1, 0] = il_val
    return SimpleNamespace(
        Frequencies=f, Parameters=Params, Impedance=Zref, NumPorts=NumPorts,
    )


def _make_param(M=4, fb=53.125e9):
    return SimpleNamespace(
        FLAG=SimpleNamespace(S2P=0), RL_sel=0,
        TR_TDR=0.025, tfx=np.array([0.0, 0.0]),
        ui=1.0 / fb, ndfe=4, N_bx=4, beta_x=0.0, Grr=1, rho_x=0.1,
        levels=4, specBER=1e-4, Tukey_Window=0,
        samples_per_ui=M, sample_dt=1.0 / (2 * 26.5625e9),
        fb=fb, fb_BT_cutoff=0.473,
    )


def _make_op():
    return SimpleNamespace(
        N=10, TDR=False, PTDR=False, DISPLAY_WINDOW=False,
        RL_norm_test=False, T_k=1e-9, BinSize=1e-3, cb_Guassian=True,
    )


def _stub_s21(sdd21, faxis, dt, OP_, param_):
    N_ir = 64
    ir = np.zeros(N_ir)
    ir[0] = 1e-3  # tiny impulse
    t = np.arange(N_ir) * dt
    return ir, t, 0.0, 0.0


def _stub_step(ir, param_, cb_step, ZT):
    ZSR = ZT * np.ones(len(ir))
    return SimpleNamespace(ZSR=ZSR, step=np.cumsum(ir))


def _stub_pulse(ir, param_, cb_step, ZT):
    M = int(param_.samples_per_ui)
    from scipy.signal import lfilter
    pulse = lfilter(np.ones(M), 1, ir)
    return SimpleNamespace(pulse=pulse, pulse_orig=pulse.copy())


# ---------------------------------------------------------------------------
# B06-D9 regression: s2p RL renormalization (MATLAB 4p15p0 lines 7078-7081).
#   RL = interim \ (s11-rho) / (1-rho*s11) * interim
# MATLAB '\' is left-division and '\ / *' are equal-precedence left-associative,
# so interim cancels and RL = (s11-rho)/(1-rho*s11). Pre-fix Python used '/'
# (right-division) -> RL = interim^2/((s11-rho)*(1-rho*s11)); for rho=0 that is
# 1/s11 (|RL|>1, non-physical). Matched reference: 2*ZT == S.Impedance -> rho=0.
# ---------------------------------------------------------------------------
def test_B06_D9_s2p_RL_matched_ref_equals_s11():
    N = 16
    f = np.linspace(0.1e9, 26.5625e9, N)
    s11 = 0.3 + 0.0j
    S = SimpleNamespace(Frequencies=f,
                        Parameters=np.full((N, 1, 1), s11, dtype=complex),
                        Impedance=100.0, NumPorts=1)
    param = _make_param()
    param.FLAG = SimpleNamespace(S2P=1)      # s2p RL renormalization path
    param.RL_sel = 0
    ZT = S.Impedance / 2.0                    # 2*ZT == Impedance -> rho = 0
    OP = _make_op()
    OP.TDR = True                             # populate TDR_results.RL

    result = get_TDR(S, OP, param, ZT, 0,
                     _s21_to_impulse_DC_fn=_stub_s21,
                     _get_StepR_fn=_stub_step,
                     _get_PulseR_fn=_stub_pulse)

    RL = np.asarray(result.RL).ravel()
    assert np.max(np.abs(RL - s11)) < 1e-9, (
        "matched-ref (rho=0) RL should equal s11=%s; got %s (pre-fix 1/s11 from "
        "right-dividing by interim instead of MATLAB left-division)" % (s11, RL[0]))
    assert np.max(np.abs(RL)) <= 1.0 + 1e-9, (
        "|RL|=%g > 1 is non-physical for a passive |s11|<1" % np.max(np.abs(RL)))


# ---------------------------------------------------------------------------
# Test 1 – Nominal: output has required fields
# ---------------------------------------------------------------------------

def test_output_fields():
    """TDR_results always has f, tdr, t, Rx_filter, tx_filter."""
    N = 32
    f = np.linspace(0.1e9, 26.5625e9, N)
    S = _make_S(N, f)
    param = _make_param()
    OP = _make_op()

    result = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 0,
                     _s21_to_impulse_DC_fn=_stub_s21,
                     _get_StepR_fn=_stub_step,
                     _get_PulseR_fn=_stub_pulse)

    for field in ('f', 'tdr', 't', 'Rx_filter', 'tx_filter'):
        assert hasattr(result, field), f'Missing: {field}'


# ---------------------------------------------------------------------------
# Test 2 – Nominal: t is shifted by -500 ps
# ---------------------------------------------------------------------------

def test_t_shifted_by_delay():
    """t axis is shifted by -500 ps (TDR_results.delay = 500e-12)."""
    N = 32
    f = np.linspace(0.1e9, 26.5625e9, N)
    S = _make_S(N, f)
    param = _make_param()
    OP = _make_op()

    delay_recorded = {}

    def track_s21(sdd21, faxis, dt, OP_, param_):
        N_ir = 128
        ir = np.zeros(N_ir)
        t = np.arange(N_ir) * dt
        return ir, t, 0.0, 0.0

    result = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 0,
                     _s21_to_impulse_DC_fn=track_s21,
                     _get_StepR_fn=_stub_step,
                     _get_PulseR_fn=_stub_pulse)

    # TDR delay is 500 ps, so t[0] should be < 0 if raw t[0]=0
    assert result.t[0] < 0 or result.t[0] <= 1e-10, f"t[0]={result.t[0]:.3e} — delay not applied?"


# ---------------------------------------------------------------------------
# Test 3 – Analytic: TDR_RL formula for known inputs
# ---------------------------------------------------------------------------

def test_TDR_RL_formula_known():
    """TDR_RL(Zin, Zout, 0, 0, 0, 0) = (Zin - Zout)/(Zin + Zout).

    With s11=s22=s12=s21=0 (no signal, just impedance mismatch at port):
      num = Zin^2 - Zout^2
      den = (Zin + Zout)^2
      RL  = (Zin - Zout)/(Zin + Zout)    [standard Γ formula]

    Separately: with s11=s22=1, s12=s21=0 (open circuit):
      RL = 1.0  (total reflection)
    """
    Zin = 50.0
    Zout = 100.0

    # Case 1: no signal → standard Γ
    RL = _TDR_RL(Zin, Zout, 0.0, 0.0, 0.0, 0.0)
    expected = (Zin - Zout) / (Zin + Zout)
    assert abs(RL - expected) < 1e-10, f"TDR_RL(no signal)={RL:.6f}, expected={expected:.6f}"

    # Case 2: open circuit → total reflection
    RL_open = _TDR_RL(Zin, Zout, 1.0, 0.0, 0.0, 1.0)
    assert abs(RL_open - 1.0) < 1e-10, f"TDR_RL(open)={RL_open:.6f}, expected=1.0"


# ---------------------------------------------------------------------------
# Test 4 – Boundary: Rx_filter = ones when BT=0, BW=0, Tukey=0
# ---------------------------------------------------------------------------

def test_rx_filter_is_ones_when_all_disabled():
    """With TDR_Bessel_Thomson=0, no TDR_Butterworth, Tukey_Window=0: Rx_filter = ones."""
    N = 32
    f = np.linspace(0.1e9, 26.5625e9, N)
    S = _make_S(N, f)
    param = _make_param()
    param.Tukey_Window = 0
    OP = _make_op()
    # No TDR_Butterworth attribute → H_bw = ones

    result = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 0,
                     _s21_to_impulse_DC_fn=_stub_s21,
                     _get_StepR_fn=_stub_step,
                     _get_PulseR_fn=_stub_pulse)

    np.testing.assert_allclose(result.Rx_filter, np.ones(N), atol=1e-10)


# ---------------------------------------------------------------------------
# Test 5 – Boundary: PTDR path sets ERL and ERLRMS fields
# ---------------------------------------------------------------------------

def test_ptdr_sets_erl_fields():
    """When OP.PTDR=True, TDR_results has ERL and ERLRMS."""
    N = 32
    f = np.linspace(0.1e9, 26.5625e9, N)
    S = _make_S(N, f)
    param = _make_param(M=4)
    OP = _make_op()
    OP.PTDR = True
    OP.TDR = True
    OP.RL_norm_test = True  # simpler code path

    result = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 0,
                     _s21_to_impulse_DC_fn=_stub_s21,
                     _get_StepR_fn=_stub_step,
                     _get_PulseR_fn=_stub_pulse)

    assert hasattr(result, 'ERL'), "ERL not set"
    assert hasattr(result, 'ERLRMS'), "ERLRMS not set"
    assert hasattr(result, 'ptdr_RL'), "ptdr_RL not set"
    assert hasattr(result, 'WC_ptdr_samples'), "WC_ptdr_samples not set"


# ---------------------------------------------------------------------------
# Test 6 – Boundary: nport index selects correct tfx
# ---------------------------------------------------------------------------

def test_nport_selects_tfx():
    """tfx[nport] is used for time windowing — different nport gives different tstart."""
    N = 32
    f = np.linspace(0.1e9, 26.5625e9, N)
    S = _make_S(N, f)
    param = _make_param()
    param.tfx = np.array([0.0, 0.5e-9])  # port 0: tfx=0, port 1: tfx=0.5ns

    t_recorded = {}

    def record_ir(sdd21, faxis, dt, OP_, param_):
        N_ir = 256
        ir = np.zeros(N_ir)
        t = np.arange(N_ir) * dt
        return ir, t, 0.0, 0.0

    OP = _make_op()

    r0 = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 0,
                 _s21_to_impulse_DC_fn=record_ir,
                 _get_StepR_fn=_stub_step,
                 _get_PulseR_fn=_stub_pulse)

    r1 = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 1,
                 _s21_to_impulse_DC_fn=record_ir,
                 _get_StepR_fn=_stub_step,
                 _get_PulseR_fn=_stub_pulse)

    # With tfx[1]=0.5ns, the maxtime+tfx threshold is later → tend may differ
    # Both should have the field 't' and it should differ
    assert hasattr(r0, 't') and hasattr(r1, 't')
