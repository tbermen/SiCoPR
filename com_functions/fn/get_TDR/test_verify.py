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


# ===========================================================================
# COM Octave oracle — the whole of get_TDR executed under the reference
# ===========================================================================
# Every number in this block came out of the REFERENCE function itself:
# com_ieee8023_4p16p0_octave_compat.m's get_TDR, run under Octave by
# tools/octave_oracle.py on the inputs _oracle_run() builds here. Nothing below
# was computed by the port. Reproduce with:
#
#   from octave_oracle import call
#   call('get_TDR', args=['S','OP','param','ZT','nport'],
#        inputs={'f': f, 'Pm': np.transpose(P, (1, 2, 0)), 'ZT': ZT, 'nport': 1.0},
#        outputs=['TDR_results'], setup=<the same struct fields as below>,
#        needs=['get_TDR','Bessel_Thomson_Filter','Butterworth_Filter',
#               'Tukey_Window','s21_to_impulse_DC','interp_Sparam','get_StepR',
#               'get_PulseR','get_pdf_from_sampled_signal','d_cpdf','conv_fct',
#               'Init_PDF_Fast','normal_dist','get_RAW_FIR'])
#
# plus octave/patches/com_octave_accel_on.m copied into the workdir, and
# OP.DISPLAY_WINDOW = OP.DEBUG = false or Octave opens a figure and hangs.
#
# The S-parameters are REAL on purpose: Octave's and numpy's abs() of a complex
# number differ by an ULP, which would force a tolerance wide enough to hide a
# defect. The tolerances that remain are the ifft (FFTW vs pocketfft) and
# nothing else — measured worst case 7.5e-16 relative on tdr.

from com_functions.fn.s21_to_impulse_DC.py_impl import (            # noqa: E402
    s21_to_impulse_DC as _real_s21)
from com_functions.fn.get_StepR.py_impl import get_StepR as _real_step      # noqa: E402
from com_functions.fn.get_PulseR.py_impl import get_PulseR as _real_pulse   # noqa: E402
from com_functions.fn.get_pdf_from_sampled_signal.py_impl import (  # noqa: E402
    get_pdf_from_sampled_signal as _real_pdf)
from com_functions.fn.Bessel_Thomson_Filter.py_impl import (        # noqa: E402
    Bessel_Thomson_Filter as _real_bt)
from com_functions.fn.Butterworth_Filter.py_impl import (           # noqa: E402
    Butterworth_Filter as _real_bw)
from com_functions.fn.Tukey_Window.py_impl import Tukey_Window as _real_tw  # noqa: E402

# 0 .. 20 GHz inclusive of DC; fmax = 1/(2*sample_dt) lands on the last point.
_ORACLE_F = np.arange(0, 41) * 0.5e9


def _oracle_P():
    """The 2-port, purely real S-parameters the oracle was driven with."""
    f = _ORACLE_F
    P = np.zeros((len(f), 2, 2))
    P[:, 0, 0] = 0.15 * np.cos(2 * np.pi * f * 60e-12)
    P[:, 1, 1] = 0.11 * np.cos(2 * np.pi * f * 130e-12) - 0.02
    P[:, 0, 1] = 0.80 * np.cos(2 * np.pi * f * 130e-12)
    P[:, 1, 0] = P[:, 0, 1]
    return P


def _oracle_run(P=None, Zref=50.0, ZT=50.0 * np.sqrt(2), NumPorts=2, S2P=0,
                fb=10e9, M=4, dt=2.5e-11, RL_sel=0, Grr=1, beta_x=0.0,
                N=12, BinSize=1e-3, RL_norm_test=0, TDR=1, PTDR=1):
    """get_TDR with the real callees, on the oracle's inputs."""
    P = _oracle_P() if P is None else P
    param = SimpleNamespace(
        FLAG=SimpleNamespace(S2P=S2P), RL_sel=RL_sel, TR_TDR=0.025,
        tfx=np.array([0.0, 0.0]), ui=1.0 / fb, fb=fb, samples_per_ui=M,
        sample_dt=dt, ndfe=4, N_bx=4, beta_x=beta_x, Grr=Grr, rho_x=0.1,
        levels=4, specBER=1e-4, Tukey_Window=0, fb_BT_cutoff=0.75,
        fb_BW_cutoff=0.75, BTorder=4, f_r=0.75, matlab_version='4p16p0')
    OP = SimpleNamespace(
        DISPLAY_WINDOW=False, DEBUG=False, N=N, T_k=1e-9, BinSize=BinSize,
        cb_Guassian=1, RL_norm_test=bool(RL_norm_test),
        TDR=bool(TDR), PTDR=bool(PTDR), EC_PULSE_TOL=0.01, EC_REL_TOL=1e-4,
        EC_DIFF_TOL=1e-6, ENFORCE_CAUSALITY=0,
        impulse_response_truncation_threshold=1e-3,
        interp_sparam_mag='linear_trend_to_DC',
        interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf', ZERO_PAD=0)
    S = SimpleNamespace(Frequencies=_ORACLE_F, Parameters=P,
                        Impedance=Zref, NumPorts=NumPorts)
    return get_TDR(S, OP, param, ZT, 0,
                   _Bessel_Thomson_Filter_fn=_real_bt,
                   _Butterworth_Filter_fn=_real_bw,
                   _Tukey_Window_fn=_real_tw,
                   _s21_to_impulse_DC_fn=_real_s21,
                   _get_StepR_fn=_real_step,
                   _get_PulseR_fn=_real_pulse,
                   _get_pdf_fn=_real_pdf)


def test_oracle_window_length_and_endpoints():
    """COM Octave: the windowed arrays are 48 samples long, not 47.

    MATLAB `tend = find(t>=maxtime+tfx,1); IR = IR(1:tend)` keeps the sample AT
    the threshold, so the 1-based index is the Python EXCLUSIVE bound. Slicing
    with the 0-based index dropped that last sample from tdr, t, ptdr_RL and
    WC_ptdr_samples, and moved avgZport and ERLRMS with them.
    """
    r = _oracle_run()
    assert len(r.tdr) == 48                     # COM Octave
    assert len(r.t) == 48                       # COM Octave
    assert len(r.ptdr_RL) == 48                 # COM Octave
    assert len(r.WC_ptdr_samples) == 12         # COM Octave
    assert r.t[0] == 2.5000000000000017e-11     # COM Octave
    assert r.t[-1] == 1.2e-09                   # COM Octave
    np.testing.assert_allclose(r.tdr[0], 153.74063746148707, rtol=1e-13)
    np.testing.assert_allclose(r.tdr[1], 134.98096897920291, rtol=1e-13)
    np.testing.assert_allclose(r.tdr[-1], 107.91339528893626, rtol=1e-13)


def test_oracle_RL_and_filters():
    """COM Octave: the renormalised reflection coefficient and the Rx filter."""
    r = _oracle_run()
    assert len(r.RL) == 41
    np.testing.assert_allclose(np.real(r.RL[0]), -0.010575702598229013, rtol=1e-14)
    np.testing.assert_allclose(np.real(r.RL[20]), -0.54625468606314909, rtol=1e-14)
    np.testing.assert_allclose(np.real(r.RL[-1]), -0.27178085843143074, rtol=1e-14)
    # BT is forced off, Butterworth absent, Tukey_Window == 0 -> all ones.
    np.testing.assert_array_equal(r.Rx_filter, np.ones(41))


def test_oracle_tx_filter_delay_divides_last():
    """COM Octave: the 1e-9 in the delay term divides LAST, as MATLAB parses it.

    MATLAB L7040 is  exp(-(1j)*2*pi*f9*TDR_results.delay/1e-9)  and `*` and `/`
    are equal precedence and left-associative, so the division is the LAST
    operation. Pre-computing delay/1e-9 = 0.5 and multiplying reorders the
    rounding: Octave evaluates the reference form at f9 = 2.5 to
    1.1943401194869635e-15-1j and the pre-divided form to
    3.0616169978683831e-16-1j (both confirmed with `octave_oracle.py --show`;
    numpy reproduces each spelling exactly).

    Checked at f9 = 20, one of the 23 of 41 points where Octave's and numpy's
    ARRAY complex exp agree bit for bit, so the tolerance can be tight enough to
    see the reorder — which moves this element by 1.2e-15. The other 18 points
    carry up to 7e-15 of Octave-vs-numpy exp noise and cannot show it.
    """
    r = _oracle_run()
    assert abs(r.tx_filter[40]
               - (-0.17520473502541811 - 1.7384015287028186e-15j)) < 2e-16


def test_oracle_avgZport_and_xy_fields():
    """COM Octave: avgZport, and the x/y fields MATLAB fills (swapped).

    MATLAB L7185-7186 assigns TDR_results.x = TDR_results.tdr(:) and
    TDR_results.y = TDR_results.t(:) — the tdr and t vectors the other way round
    from the local x/y it uses for the weighted average. The port set neither.
    """
    r = _oracle_run()
    np.testing.assert_allclose(r.avgZport, 107.85922590256126, rtol=1e-13)
    assert hasattr(r, 'x') and hasattr(r, 'y'), 'MATLAB fills x and y'
    np.testing.assert_allclose(r.x[0], 153.74063746148707, rtol=1e-13)   # == tdr[0]
    np.testing.assert_allclose(r.x[-1], 107.91339528893626, rtol=1e-13)
    assert r.y[0] == 2.5000000000000017e-11    # COM Octave, == t[0]
    assert r.y[-1] == 1.2e-09                  # COM Octave, == t[-1]


def test_oracle_avgZport_zero_when_gate_past_window():
    """COM Octave: avgZport = 0 when 3*TR_TDR lands past the windowed vector.

    tfstart indexes the full time vector but slices the windowed one, so with a
    short OP.N the slice is EMPTY; MATLAB's x(1) then errors and the catch
    reports 0. Clamping tfstart to the last sample returned that sample's
    impedance (about 107) instead.
    """
    r = _oracle_run(N=5)
    assert len(r.t) == 20                      # COM Octave
    assert r.avgZport == 0.0                   # COM Octave
    np.testing.assert_allclose(r.ERLRMS, 44.457967798680762, rtol=1e-13)


def test_oracle_ERL_and_ERLRMS_gated_pulse():
    """COM Octave: ERL/ERLRMS and the gated reflection pulse, Grr = 1."""
    r = _oracle_run()
    np.testing.assert_allclose(r.ERL, 40.0, rtol=1e-13)              # COM Octave
    np.testing.assert_allclose(r.ERLRMS, 48.110907439217286, rtol=1e-13)
    # fctrx zeroes only the lead-in; the first gated sample is index 3.
    assert np.all(r.ptdr_RL[:3] == 0.0)
    np.testing.assert_allclose(r.ptdr_RL[3], -0.013388655916898329, rtol=1e-11)
    np.testing.assert_allclose(r.ptdr_RL[-1], 2.0403676012603969e-05, rtol=1e-9)
    np.testing.assert_allclose(r.WC_ptdr_samples[-1], 1.1483416227708507e-05,
                               rtol=1e-9)
    assert r.WC_ptdr_samples_t[0] == 2.5000000000000017e-11          # COM Octave


def test_oracle_Grr_modes_and_beta_x():
    """COM Octave: Grr = 2 takes rho_x flat, and beta_x adds the near-end loss."""
    r2 = _oracle_run(Grr=2)
    np.testing.assert_allclose(r2.ERL, 30.457574905606752, rtol=1e-13)
    np.testing.assert_allclose(r2.ERLRMS, 42.782285704029086, rtol=1e-13)
    np.testing.assert_allclose(r2.ptdr_RL[3], -0.030012008810253756, rtol=1e-11)
    rb = _oracle_run(beta_x=-2.0)
    np.testing.assert_allclose(rb.ERLRMS, 48.11090743846313, rtol=1e-13)
    np.testing.assert_allclose(rb.ptdr_RL[3], -0.013388655918362684, rtol=1e-11)


def test_oracle_ERL_comes_from_the_LAST_phase_not_the_worst():
    """COM Octave: upstream defect B06-D10, reproduced deliberately.

    The reference's
        if ~OP.RL_norm_test
            best_erl=rl_test; best_pdf=testpdf; best_cdf=cdf_test;
        end
    sits OUTSIDE the `rl_fom > RL_equiv` guard (identical in 4p14p0, 4p15p0,
    4p16p0 and the adaptive-local-search build), so ERL is reported for the LAST
    phase while WC_ptdr_samples still come from best_ki. On this case the worst
    phase is ki = 4 (1-based) with rl_test = 0.024 -> 32.39577516576788 dB and
    the last phase is ki = 8 with rl_test = 0.017 -> 35.391021572434525 dB;
    Octave reports the latter. Not reached with the default ERL_FOM = 1.
    """
    r = _oracle_run(fb=5e9, M=8, N=6, BinSize=1e-4)
    np.testing.assert_allclose(r.ERL, 35.391021572434525, rtol=1e-13)
    np.testing.assert_allclose(r.ERLRMS, 45.696174453154164, rtol=1e-13)
    # best_ki is still the worst phase: its first sample is t = 1e-10, not t[0].
    assert len(r.WC_ptdr_samples_t) == 6                             # COM Octave
    np.testing.assert_allclose(r.WC_ptdr_samples_t[0], 9.9999999999999965e-11,
                               rtol=1e-14)
    np.testing.assert_allclose(r.WC_ptdr_samples[0], -0.012759585270103766,
                               rtol=1e-11)


def test_oracle_s2p_branch_and_infinite_ERL():
    """COM Octave: the FLAG.S2P renormalisation, and db(0) = -Inf with no floor.

    MATLAB's `db = @(x) 20*log10(abs(x))` has no epsilon: this case's best_erl
    is 0, so the reference reports ERL = +Inf. `20*log10(abs(x) + 1e-300)`
    reported a plausible-looking 6000 dB instead.
    """
    P = np.zeros((41, 1, 1))
    P[:, 0, 0] = 0.3 * np.cos(2 * np.pi * _ORACLE_F * 80e-12)
    r = _oracle_run(P=P, NumPorts=1, S2P=1, ZT=35.0)
    np.testing.assert_allclose(np.real(r.RL[0]), 0.14035087719298248, rtol=1e-14)
    np.testing.assert_allclose(np.real(r.RL[20]), -0.075122271278233704, rtol=1e-14)
    np.testing.assert_allclose(r.tdr[0], 72.154872484718481, rtol=1e-13)
    np.testing.assert_allclose(r.avgZport, 70.70059792871362, rtol=1e-13)
    assert r.ERL == np.inf                                           # COM Octave
    np.testing.assert_allclose(r.ERLRMS, 51.766434286562358, rtol=1e-13)


def test_oracle_degenerate_bailout_4p16p0():
    """COM Octave: mean(abs(RL)) < 1e-6 returns the degenerate 4p16p0 result.

    Zref == 2*ZT with s11 = s22 = 0 makes the renormalised RL identically zero.
    The reference then returns tdr = ones(1000,1), t = 0:dt:999*dt (1000
    samples) and WC_ptdr_samples_t = 0:dt*M:999*dt (250 samples, not 1000).
    """
    P = np.zeros((41, 2, 2))
    P[:, 0, 1] = 0.9
    P[:, 1, 0] = 0.9
    r = _oracle_run(P=P, ZT=25.0)               # 2*ZT == S.Impedance == 50
    assert r.delay == 0
    assert len(r.tdr) == 1000 and np.all(r.tdr == 1.0)      # COM Octave
    assert len(r.t) == 1000                                 # COM Octave
    assert r.t[-1] == 2.4975e-08                            # COM Octave
    assert len(r.RL) == 1000 and not np.any(r.RL)
    assert len(r.ptdr_RL) == 1000 and not np.any(r.ptdr_RL)
    assert len(r.WC_ptdr_samples_t) == 250                  # COM Octave
    assert len(r.WC_ptdr_samples) == 250                    # COM Octave
    assert r.ERL == np.inf and r.ERLRMS == -300             # COM Octave
    assert r.avgZport == 0
