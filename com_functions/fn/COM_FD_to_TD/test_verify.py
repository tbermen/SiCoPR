# ============================================================
# MATLAB GROUND TRUTH
# COM_FD_to_TD converts frequency-domain sdd21 to time-domain impulse/pulse
# responses, applying Butterworth/BT filters and computing SCMR metrics.
#
# Key invariants verified analytically:
# 1. filter(ones(1,M), 1, ir) = lfilter(ones(M), 1, ir) — running sum of M samples
#    For a unit impulse ir=[1,0,0,...], pulse response = ones(M) followed by zeros
#
# 2. Amplitude scaling: uneq_imp_response *= A when USE_channel_amplitude=True
#
# 3. SCMR_CD_ch = 10*log10(P_signal / CMn_CD^2)
#    With known P_signal and CMn, this is a direct formula.
#
# 4. P_signal = norm(PR_ORIG_fltr_sampled)^2 where sampled at phase of peak
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace
from scipy.signal import lfilter

from com_functions.fn.COM_FD_to_TD.py_impl import COM_FD_to_TD


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_simple_ir(N, peak_at=0, amplitude=1.0):
    """Unit impulse at peak_at, complex-valued."""
    ir = np.zeros(N, dtype=complex)
    ir[peak_at] = amplitude
    return ir


def _make_chdata(N=64, M=4, A=1.0):
    faxis = np.linspace(0, 25e9, N)
    # Simple unit-transfer function
    sdd21 = np.ones(N, dtype=complex)
    ch = SimpleNamespace(
        faxis=faxis,
        sdd21=sdd21.copy(),
        sdd21_raw=sdd21.copy(),
        sdd21_orig=sdd21.copy(),
        scd21_orig=np.zeros(N, dtype=complex),
        sdc21_orig=np.zeros(N, dtype=complex),
        A=A,
        type='THRU',
        base='test',
    )
    return ch, faxis


def _make_param(faxis, M=4, sigma_X=1.0):
    N = len(faxis)
    return SimpleNamespace(
        samples_per_ui=M,
        sample_dt=faxis[-1] / N if len(faxis) > 1 else 1e-11,
        number_of_s4p_files=1,
        package_testcase_i=1,
        sigma_X=sigma_X,
        ndfe=4,
        f2=10e9,
        f1=0.5e9,
        levels=4,
        P_peak=1e-4,
        fb=53.125e9,
        fb_BT_cutoff=0.473,
        Z0=50.0,
    )


def _make_op():
    return SimpleNamespace(
        Bessel_Thomson=False,
        Butterworth=False,
        transmitter_transition_time=8e-3,
        RX_CALIBRATION=False,
        PSDRXCAL=False,
        DEBUG=False,
        DISPLAY_WINDOW=True,
        ENFORCE_CAUSALITY=False,
    )


# ---------------------------------------------------------------------------
# Test 1 – Nominal: running-sum filter correct for unit impulse
# ---------------------------------------------------------------------------

def test_pulse_response_from_impulse():
    """lfilter(ones(M),1,ir) of a unit impulse = ones(M) then zeros.
    MATLAB ground truth: filter(ones(1,4),1,[1,0,0,...]) = [1,1,1,1,0,0,...].
    """
    M = 4
    N = 16
    # Stub s21_fn returns a unit impulse
    def stub_s21(sdd21, faxis, dt, OP, param):
        ir = np.zeros(N)
        ir[0] = 1.0
        t = np.arange(N) * dt
        return ir, t, 0.0, 0.0

    ch, faxis = _make_chdata(N=N, M=M)
    param = _make_param(faxis, M=M)
    OP = _make_op()

    result = COM_FD_to_TD(
        [ch], param, OP,
        _s21_to_impulse_DC_fn=stub_s21,
        _Bessel_Thomson_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _Butterworth_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _get_cm_noise_fn=lambda M, pr, lev, pp, op: SimpleNamespace(CMn=0.1, CMn_peak=0.1),
    )
    pr = result[0].uneq_pulse_response
    expected = lfilter(np.ones(M), 1, np.array([1.0] + [0.0] * (N - 1)))
    np.testing.assert_allclose(pr, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 2 – Nominal: amplitude scaling applied correctly
# ---------------------------------------------------------------------------

def test_amplitude_scaling():
    """uneq_imp_response *= A when USE_channel_amplitude=True (first channel, no calibration)."""
    M = 4
    N = 16
    A = 0.5

    def stub_s21(sdd21, faxis, dt, OP, param):
        ir = np.zeros(N)
        ir[0] = 1.0
        t = np.arange(N) * dt
        return ir, t, 0.0, 0.0

    ch, faxis = _make_chdata(N=N, M=M, A=A)
    param = _make_param(faxis, M=M)
    OP = _make_op()

    result = COM_FD_to_TD(
        [ch], param, OP,
        _s21_to_impulse_DC_fn=stub_s21,
        _Bessel_Thomson_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _Butterworth_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _get_cm_noise_fn=lambda M, pr, lev, pp, op: SimpleNamespace(CMn=0.1, CMn_peak=0.1),
    )
    # The first sample of the amplitude-scaled IR should be A=0.5
    assert abs(result[0].uneq_imp_response[0] - A) < 1e-12, \
        f"Expected {A}, got {result[0].uneq_imp_response[0]}"


# ---------------------------------------------------------------------------
# Test 3 – Nominal: SCMR_CD_ch formula matches manual calculation
# ---------------------------------------------------------------------------

def test_scmr_cd_ch_formula():
    """SCMR_CD_ch = 10*log10(P_signal / CMn^2). Manual: known P_signal, CMn."""
    M = 4
    N = 16
    CMn_val = 0.2
    A = 1.0

    # Stub IR: unit impulse; pulse response = ones(4) then zeros
    def stub_s21(sdd21, faxis, dt, OP, param):
        ir = np.zeros(N)
        ir[0] = 1.0
        return ir, np.arange(N) * dt, 0.0, 0.0

    ch, faxis = _make_chdata(N=N, M=M, A=A)
    param = _make_param(faxis, M=M)
    OP = _make_op()

    result = COM_FD_to_TD(
        [ch], param, OP,
        _s21_to_impulse_DC_fn=stub_s21,
        _Bessel_Thomson_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _Butterworth_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _get_cm_noise_fn=lambda M, pr, lev, pp, op: SimpleNamespace(CMn=CMn_val, CMn_peak=CMn_val),
    )
    # With unit IR, pulse resp = [1,1,1,1,0,0,...]; peak at idx 3 (last of run)
    PR = lfilter(np.ones(M), 1, np.array([1.0] + [0.0] * (N - 1)))
    ipeak = int(np.argmax(PR))
    istart = ipeak % M
    nsteps = N // M
    PR_sampled = PR[istart:istart + nsteps * M:M]
    P_signal = float(np.linalg.norm(PR_sampled) ** 2)
    expected_scmr = 10.0 * np.log10(P_signal / CMn_val ** 2)

    assert abs(result[0].SCMR_CD_ch - expected_scmr) < 0.01, \
        f"Expected {expected_scmr:.3f}, got {result[0].SCMR_CD_ch:.3f}"


# ---------------------------------------------------------------------------
# Test 4 – Edge: amplitude scaling disabled for second channel with RX_CALIBRATION
# ---------------------------------------------------------------------------

def test_amplitude_not_applied_for_rx_cal_channel_2():
    """With RX_CALIBRATION=True and i>0, amplitude scaling is skipped."""
    M = 4
    N = 16
    A = 2.0

    def stub_s21(sdd21, faxis, dt, OP, param):
        ir = np.zeros(N)
        ir[0] = 1.0
        return ir, np.arange(N) * dt, 0.0, 0.0

    ch1, faxis = _make_chdata(N=N, M=M, A=1.0)
    ch2, _ = _make_chdata(N=N, M=M, A=A)  # A=2.0 but should NOT be applied
    ch2.base = 'xtalk'
    param = _make_param(faxis, M=M)
    param.number_of_s4p_files = 2
    OP = _make_op()
    OP.RX_CALIBRATION = True

    result = COM_FD_to_TD(
        [ch1, ch2], param, OP,
        _s21_to_impulse_DC_fn=stub_s21,
        _Bessel_Thomson_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _Butterworth_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _get_cm_noise_fn=lambda M, pr, lev, pp, op: SimpleNamespace(CMn=0.1, CMn_peak=0.1),
    )
    # ch2 (i=1) should NOT have amplitude scaling → IR[0]=1.0, not 2.0
    assert abs(result[1].uneq_imp_response[0] - 1.0) < 1e-12, \
        f"Expected 1.0 (no scaling), got {result[1].uneq_imp_response[0]}"
    # ch1 (i=0) SHOULD have amplitude (A=1.0)
    assert abs(result[0].uneq_imp_response[0] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Test 5 – Boundary: filters are actually applied (non-unit Butterworth)
# ---------------------------------------------------------------------------

def test_filter_applied_to_sdd21():
    """H_filters multiplies sdd21 before passing to s21_fn for raw_filtered variant."""
    M = 4
    N = 64
    # Track what sdd21 value was passed to s21_fn for the raw_filtered call
    received_sdd21 = {}

    call_count = [0]

    def stub_s21_track(sdd21, faxis, dt, OP, param):
        call_count[0] += 1
        # 3rd call = raw_filtered (sdd21_raw * H_filters)
        if call_count[0] == 3:
            received_sdd21['val'] = sdd21.copy()
        ir = np.zeros(N)
        ir[0] = 1.0
        return ir, np.arange(N) * dt, 0.0, 0.0

    # BW filter that halves amplitude at all frequencies
    def half_bw(p, f, e):
        return 0.5 * np.ones(len(f))

    ch, faxis = _make_chdata(N=N, M=M)
    param = _make_param(faxis, M=M)
    OP = _make_op()
    OP.transmitter_transition_time = 0.0  # H_t=1

    result = COM_FD_to_TD(
        [ch], param, OP,
        _s21_to_impulse_DC_fn=stub_s21_track,
        _Bessel_Thomson_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _Butterworth_Filter_fn=half_bw,
        _get_cm_noise_fn=lambda M, pr, lev, pp, op: SimpleNamespace(CMn=0.1, CMn_peak=0.1),
    )
    # The 3rd s21_fn call (raw_filtered) should receive sdd21_raw * 0.5
    assert 'val' in received_sdd21
    np.testing.assert_allclose(
        np.abs(received_sdd21['val']),
        np.abs(ch.sdd21_raw) * 0.5,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Test 6 – Shape: all output fields exist and have correct types
# ---------------------------------------------------------------------------

def test_output_fields_exist():
    """After COM_FD_to_TD, all required chdata fields are present."""
    M = 4
    N = 16

    def stub_s21(sdd21, faxis, dt, OP, param):
        ir = np.zeros(N)
        ir[0] = 1.0
        return ir, np.arange(N) * dt, 0.0, 0.0

    ch, faxis = _make_chdata(N=N, M=M)
    param = _make_param(faxis, M=M)
    OP = _make_op()

    result = COM_FD_to_TD(
        [ch], param, OP,
        _s21_to_impulse_DC_fn=stub_s21,
        _Bessel_Thomson_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _Butterworth_Filter_fn=lambda p, f, e: np.ones(len(f)),
        _get_cm_noise_fn=lambda M, pr, lev, pp, op: SimpleNamespace(CMn=0.1, CMn_peak=0.1),
    )
    ch_out = result[0]
    required_fields = [
        'uneq_imp_response', 'uneq_pulse_response', 't',
        'uneq_imp_response_raw', 'uneq_pulse_response_raw',
        'uneq_imp_response_orig', 'uneq_pulse_response_orig',
        'uneq_pulse_response_orig_filtered',
        'uneq_CD_imp_response_filtered', 'uneq_pulse_CD_response_filtered',
        'uneq_DC_imp_response_filtered', 'uneq_pulse_DC_response_filtered',
        'P_signal', 'SCMR_CD_ch', 'SCMR_CD_ch_pk', 'SCMR_DC_ch', 'SCMR_DC_ch_pk',
        'CD_CM_RMS', 'VCM_CD_HF_struct', 'VCM_DC_HF_struct',
        'causality_correction_dB', 'truncation_dB',
    ]
    for fld in required_fields:
        assert hasattr(ch_out, fld), f'Missing field: {fld}'


# --------------------------------------------------------------------------
# SCMR on a perfectly balanced channel: no epsilon floor.
#
# ML 1342-1345 is 10*log10(V_peak^2/CMn^2) with nothing added to the
# denominator. The port had `+ 1e-300`, which turns the divide-by-zero into a
# finite number. COM Octave, on the arithmetic itself:
#
#     10*log10(1.0/0.0)            ->  inf        the reference's form
#     10*log10(1.0/(0.0+1e-300))   ->  3000       the floored form
#     10*log10(0.0/0.0)            ->  nan        the reference's form
#     10*log10(0.0/(0.0+1e-300))   -> -inf        the floored form
#
# 3000 dB reads like a measurement. Inf does not, which is the point.
#
# This is the time-domain twin of the defect found in FD_Processing's
# SCMR_FD_CD_ch_dB, whose own test fixture used scd21_orig = zeros, so that one
# was live rather than hypothetical.
# --------------------------------------------------------------------------

def test_scmr_has_no_epsilon_floor_on_a_perfectly_balanced_channel():
    """scd21_orig and sdc21_orig identically zero means no common-mode
    conversion at all, so CMn is exactly 0 and the ratio has no finite value.

    This fixture exercises BOTH forms at once: the peak fields divide a
    non-zero V_peak by zero and must be +inf, while the average fields divide a
    P_signal that is also zero and must be nan. The floored version turns those
    into 3000 and -inf respectively, so either substitution is caught.
    """
    ch, faxis = _make_chdata()
    ch.scd21_orig = np.zeros(len(faxis), dtype=complex)
    ch.sdc21_orig = np.zeros(len(faxis), dtype=complex)

    # Inject the REAL dependencies, as sicopr's _wired_COM_FD_to_TD does.
    # Called bare this runs the module's stubs, and the get_cm_noise stub
    # returns CMn = RMS(pulse_resp) rather than the true common-mode noise, so
    # the denominator is never zero and the test reads 599.8 dB instead. That
    # is precisely the trap tests/test_stub_reachability.py exists to catch,
    # walked into while writing a test about a different floor.
    import sicopr
    # _make_op() was written for the stub and lacks the fields the real
    # s21_to_impulse_DC reads, which is itself a sign of how long the stub has
    # stood in. Extend it locally rather than change what other tests drive.
    op = _make_op()
    op.EC_PULSE_TOL = 0.05
    op.EC_REL_TOL = 1e-3
    op.EC_DIFF_TOL = 1e-5
    op.impulse_response_truncation_threshold = 1e-3
    op.interp_sparam_mag = 'linear_trend_to_DC'
    op.interp_sparam_phase = 'extrap_cubic_to_dc_linear_to_inf'
    result = COM_FD_to_TD(
        [ch], _make_param(faxis), op,
        _s21_to_impulse_DC_fn=sicopr.s21_to_impulse_DC,
        _Bessel_Thomson_Filter_fn=sicopr.Bessel_Thomson_Filter,
        _Butterworth_Filter_fn=sicopr.Butterworth_Filter,
        _get_cm_noise_fn=sicopr.get_cm_noise)
    r0 = result[0]

    assert float(r0.VCM_CD_HF_struct.CMn) == 0.0, (
        'fixture no longer gives a zero common-mode term, so it cannot '
        'distinguish the floor; CMn = %r' % r0.VCM_CD_HF_struct.CMn)

    for name in ('SCMR_CD_ch_pk', 'SCMR_DC_ch_pk'):
        v = float(np.ravel(np.asarray(getattr(r0, name)))[0])
        assert np.isinf(v) and v > 0, (
            '%s is %.17g; COM Octave gives inf for 10*log10(x/0), and 3000 dB '
            'for the floored form, which is indistinguishable from data.'
            % (name, v))
    for name in ('SCMR_CD_ch', 'SCMR_DC_ch'):
        v = float(np.ravel(np.asarray(getattr(r0, name)))[0])
        assert np.isnan(v), (
            '%s is %.17g; COM Octave gives nan for 10*log10(0/0), and -inf '
            'for the floored form.' % (name, v))
