"""Verification tests for s21_to_impulse_DC().

# ============================================================
# MATLAB GROUND TRUTH (lines 11076-11146)
# Converts frequency-domain IL to time-domain impulse response.
# All-zero IL → eps-valued response (no interp needed).
# Non-zero IL → interp_Sparam + alternating-projections causality enforcement.
# Returns (voltage, t_base, causality_correction_dB, truncation_dB).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.s21_to_impulse_DC.py_impl import s21_to_impulse_DC, _interp_Sparam


def _op():
    return SimpleNamespace(
        interp_sparam_mag='old',
        interp_sparam_phase='old',
        EC_PULSE_TOL=0.01,
        EC_REL_TOL=1e-4,
        EC_DIFF_TOL=1e-6,
        ENFORCE_CAUSALITY=True,
        impulse_response_truncation_threshold=1e-3,
        DEBUG=True,
        ZERO_PAD=False,
    )


def _param():
    return SimpleNamespace()


def _freq_array(N=100, fmax=50e9):
    return np.linspace(0, fmax, N)


def test_zero_il_returns_four_outputs():
    """All-zero IL returns a 4-tuple."""
    freq = _freq_array()
    IL = np.zeros(len(freq))
    result = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), _op(), _param())
    assert len(result) == 4


def test_zero_il_voltage_is_array():
    """All-zero IL returns voltage as 1-D array."""
    freq = _freq_array()
    IL = np.zeros(len(freq))
    voltage, t_base, _, _ = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), _op(), _param())
    assert isinstance(voltage, np.ndarray)
    assert voltage.ndim == 1


def test_zero_il_t_base_starts_at_zero():
    """t_base[0] == 0."""
    freq = _freq_array()
    IL = np.zeros(len(freq))
    _, t_base, _, _ = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), _op(), _param())
    assert t_base[0] == pytest.approx(0.0)


def test_nonzero_il_returns_valid_result():
    """Non-zero IL (flat attenuation) returns finite voltage array."""
    freq = _freq_array()
    phase = -2 * np.pi * freq * 100e-12
    IL = 0.8 * np.exp(1j * phase)
    voltage, t_base, cc_dB, trunc_dB = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), _op(), _param())
    assert isinstance(voltage, np.ndarray)
    assert voltage.ndim == 1
    assert len(voltage) > 0
    assert np.all(np.isfinite(voltage))


def test_nonzero_il_t_base_monotonic():
    """t_base from non-zero IL is monotonically increasing."""
    freq = _freq_array()
    phase = -2 * np.pi * freq * 50e-12
    IL = 0.7 * np.exp(1j * phase)
    _, t_base, _, _ = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), _op(), _param())
    assert np.all(np.diff(t_base) >= 0)


def test_causality_correction_dB_type():
    """causality_correction_dB is a float scalar."""
    freq = _freq_array()
    phase = -2 * np.pi * freq * 80e-12
    IL = 0.9 * np.exp(1j * phase)
    _, _, cc_dB, trunc_dB = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), _op(), _param())
    assert np.isscalar(cc_dB) or (isinstance(cc_dB, np.ndarray) and cc_dB.ndim == 0)


def test_enforce_causality_false_uses_original():
    """ENFORCE_CAUSALITY=False skips projection; voltage is still finite."""
    op = _op()
    op.ENFORCE_CAUSALITY = False
    freq = _freq_array()
    phase = -2 * np.pi * freq * 50e-12
    IL = 0.8 * np.exp(1j * phase)
    voltage, _, _, _ = s21_to_impulse_DC(IL, freq, 1.0 / (2 * 50e9), op, _param())
    assert np.all(np.isfinite(voltage))


# ---------------------------------------------------------------------------
# B03-D6 / B03-D7 regression tests (MATLAB 4p15p0 lines 11229-11285).
# MATLAB causality window: impulse_response(1:start_ind)=0 (1-based, INCLUSIVE),
# impulse_response(floor(L/2):end)=0, err=max(delta)/max(impulse_response) (signed).
# MATLAB fout grid: fout=0:1/round(fmax/freq_step)*fmax:fmax (round=half-away).
# These assert MATLAB-faithful behavior; they FAIL on the pre-fix code.
# ---------------------------------------------------------------------------
def _op_lossy():
    op = _op()
    op.interp_sparam_mag = 'old'
    op.interp_sparam_phase = 'old'
    return op


def test_B03_D6_causality_leading_window_zeros_start_ind():
    """MATLAB zeros impulse_response(1:start_ind) inclusive; 0-based that is
    indices 0..start_ind (start_ind = candidates[0]). So voltage[start_ind] must
    be 0 after enforcement. Pre-fix Python zeros only [:start_ind], leaving
    voltage[start_ind] (the first above-tolerance sample) nonzero.

    Grid uses fmax/freq_step = 100.3 (far from a half-integer) so round==half-away
    and the fout grid is identical pre/post B03-D7 fix, isolating the window bug.
    EC_REL_TOL is set huge to break after one projection so the returned response
    IS the zeroed one.
    """
    freq_step = 1e9
    freq = np.arange(0, 60) * freq_step            # 0..59 GHz, freq_step=1 GHz
    fmax = 100.3e9                                   # fmax/freq_step = 100.3
    time_step = 1.0 / (2 * fmax)
    # lossy, delayed channel -> clean causal-ish pulse with a defined onset
    IL = 0.9 * np.exp(-freq / 80e9) * np.exp(-1j * 2 * np.pi * freq * 200e-12)

    op = _op_lossy()
    op.ENFORCE_CAUSALITY = True
    op.EC_REL_TOL = 1e12                             # force break after 1 projection
    param = _param()

    # Replicate the function's impulse_response to obtain start_ind and L.
    n_steps = max(1, round(fmax / freq_step))        # 100 (round==half-away here)
    fout = np.arange(0, n_steps + 1) * (fmax / n_steps)
    IL_i = _interp_Sparam(IL, freq, fout, 'old', 'old', op, param)
    IL_sym = np.concatenate([[np.real(IL_i[0])], IL_i[1:-1],
                             [np.real(IL_i[-1])], np.conj(IL_i[1:-1])[::-1]])
    ir = np.real(np.fft.ifft(IL_sym))
    half = len(ir) // 2
    m = np.max(np.abs(ir[:half]))
    start_ind = int(np.where(np.abs(ir[:half]) > m * op.EC_PULSE_TOL)[0][0])

    voltage, _, _, _ = s21_to_impulse_DC(IL, freq, time_step, op, param)

    assert start_ind < len(voltage), "start_ind must lie within the returned voltage"
    # MATLAB zeros index start_ind (0-based) as part of (1:start_ind); pre-fix leaves it nonzero.
    assert abs(voltage[start_ind]) < 1e-12, (
        "voltage[start_ind]=%g should be 0 (MATLAB zeros impulse_response(1:start_ind) "
        "inclusive)" % voltage[start_ind])


def test_B02_D4_inlined_interp_mag_floor_uses_machine_eps():
    """s21_to_impulse_DC inlines _interp_Sparam, which carries the same B02-D4
    magnitude-floor bug (floors at tiny, not machine eps). MATLAB line 8091 uses
    machine eps (2.2204e-16). Fix reaches the mainline FD->TD path via this copy.
    """
    fin = np.linspace(1e9, 20e9, 40)
    Sin = 1e-18 * np.exp(-1j * 2 * np.pi * fin * 100e-12)
    op = SimpleNamespace(DEBUG=True, ZERO_PAD=False)
    param = SimpleNamespace(fb=28e9, f_r=0.75)
    Sout = _interp_Sparam(Sin, fin, fin, 'old', 'old', op, param)
    assert np.min(np.abs(Sout)) >= 0.9 * np.finfo(float).eps, (
        "inlined _interp_Sparam min|Sout|=%g; should floor to machine eps "
        "(MATLAB 8091)" % np.min(np.abs(Sout)))


def test_B02_D5_inlined_interp_phase_extrapolates_linearly():
    """The inlined _interp_Sparam clamps phase beyond fin[-1] (B02-D5); MATLAB
    line 8185 extrapolates linearly. Fix reaches the mainline via this copy.
    """
    delay = 100e-12
    fin = np.linspace(1e9, 20e9, 40)
    fout = np.linspace(1e9, 40e9, 80)
    Sin = np.exp(-1j * 2 * np.pi * fin * delay)
    op = SimpleNamespace(DEBUG=True, ZERO_PAD=False)
    param = SimpleNamespace(fb=28e9, f_r=0.75)
    Sout = _interp_Sparam(Sin, fin, fout, 'old', 'old', op, param)
    ph = np.unwrap(np.angle(Sout))
    gd = -np.diff(ph) / np.diff(fout)
    extrap = fout[1:] > fin[-1]
    assert np.mean(gd[extrap]) > 0.5 * delay, (
        "inlined _interp_Sparam extrap group delay = %g s; expected ~%g s "
        "(MATLAB 8185 linear extrap)" % (np.mean(gd[extrap]), delay))


def test_B03_D7_fout_grid_uses_half_away_round():
    """MATLAB fout step = 1/round(fmax/freq_step)*fmax with half-away round.
    At an exact half-integer ratio (200.5) MATLAB round=201, banker's round=200.
    The symmetric impulse length is L = 2*n_steps, so with ENFORCE off and a zero
    truncation threshold, len(voltage) == 2*201 = 402 after the fix (was 400).
    """
    freq_step = 1e9
    freq = np.arange(0, 60) * freq_step
    fmax = 200.5e9                                   # fmax/freq_step = 200.5 exactly
    time_step = 1.0 / (2 * fmax)
    IL = 0.9 * np.exp(-freq / 80e9) * np.exp(-1j * 2 * np.pi * freq * 100e-12)

    op = _op_lossy()
    op.ENFORCE_CAUSALITY = False
    op.impulse_response_truncation_threshold = 0.0   # keep the full response
    voltage, _, _, _ = s21_to_impulse_DC(IL, freq, time_step, op, _param())

    assert len(voltage) == 2 * 201, (
        "len(voltage)=%d; expected 2*round(200.5)=402 (half-away). Pre-fix banker's "
        "round(200.5)=200 gives 400." % len(voltage))


# --------------------------------------------------------------------------
# The configured default path, against COM Octave itself.
#
# Every shipped configuration workbook selects interp_sparam_mag =
# 'linear_trend_to_DC' and interp_sparam_phase =
# 'extrap_cubic_to_dc_linear_to_inf'. Until 2026-09-22 no test in this file
# passed either, so the branch every run takes was the one branch never
# exercised -- and all three `std` calls live in it. That is how the
# N-vs-N-1 normalisation defect (SiCoPR d5bff6c) survived: the function was
# tested, its default path was not.
#
# The expected values below are COM Octave's own s21_to_impulse_DC, extracted
# verbatim from octave/com_ieee8023_4p16p0_octave_compat.m and run on the
# channel _default_path_channel() rebuilds. A reading of the MATLAB cannot
# catch a library-default divergence; running it can.
# --------------------------------------------------------------------------

_OCTAVE_N = 1032
_OCTAVE_ARGMAX0 = 600                       # 0-based; Octave reports 601
_OCTAVE_MAXABS = 0.65393863838784316
_OCTAVE_CAUS_DB = -13.663430588600212
_OCTAVE_TRUNC_DB = -39.535880675055282


def _default_path_channel():
    """A synthetic channel that starts at 10 MHz, so the DC-extrapolation
    branch runs. Deterministic: numpy's PCG64 stream is stable."""
    fin = np.arange(1, 4001) * 10e6
    rng = np.random.default_rng(5)
    f_ghz = fin / 1e9
    mag = 10 ** (-(0.4 * np.sqrt(f_ghz) + 0.05 * f_ghz) / 20)
    ph = -2 * np.pi * fin * 5e-9 + 2e-3 * rng.standard_normal(fin.size)
    return mag * np.exp(1j * ph), fin, 1.0 / 120e9


def _op_default_path():
    return SimpleNamespace(
        interp_sparam_mag='linear_trend_to_DC',
        interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf',
        DEBUG=0, ZERO_PAD=0, ENFORCE_CAUSALITY=1,
        EC_PULSE_TOL=0.05, EC_REL_TOL=1e-3, EC_DIFF_TOL=1e-5,
        impulse_response_truncation_threshold=1e-3)


def test_default_option_path_matches_com_octave():
    """The options every workbook selects, checked against the reference code."""
    IL, fin, ts = _default_path_channel()
    v, _t, caus, trunc = s21_to_impulse_DC(IL, fin, ts, _op_default_path(), _param())
    v = np.asarray(v).ravel()

    assert v.size == _OCTAVE_N, 'length %d, COM Octave gives %d' % (v.size, _OCTAVE_N)
    assert int(np.argmax(np.abs(v))) == _OCTAVE_ARGMAX0, (
        'peak at sample %d, COM Octave puts it at %d'
        % (int(np.argmax(np.abs(v))), _OCTAVE_ARGMAX0))
    for got, want, name in ((np.max(np.abs(v)), _OCTAVE_MAXABS, 'peak'),
                            (caus, _OCTAVE_CAUS_DB, 'causality_correction_dB'),
                            (trunc, _OCTAVE_TRUNC_DB, 'truncation_dB')):
        rel = abs(float(got) - want) / abs(want)
        assert rel < 1e-12, '%s is %.17g, COM Octave gives %.17g (rel %.2e)' % (
            name, float(got), want, rel)


def test_default_path_is_sensitive_to_std_normalisation():
    """Guard the guard. If this channel stopped exercising the outlier
    threshold, the test above would pass whichever std convention was used."""
    IL, fin, _ = _default_path_channel()
    gd = -np.diff(np.unwrap(np.angle(IL))) / np.diff(fin)
    lf = gd[:50]
    m = np.median(lf)
    kept = [np.abs(lf - m) < np.std(lf, ddof=d) for d in (0, 1)]
    assert not np.array_equal(kept[0], kept[1]), (
        'this channel no longer distinguishes N from N-1 normalisation; '
        're-craft it before trusting the oracle comparison above')
