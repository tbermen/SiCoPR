"""Verification tests for interp_Sparam().

# ============================================================
# MATLAB GROUND TRUTH (lines 7950-8165)
# Interpolates S-parameters Sin(fin) → Sout(fout).
# H_mag = abs(Sin), H_ph = unwrap(angle(Sin)).
# Anti-causal check: mean(diff(H_ph)) > 0 → warn/error.
# Magnitude methods: linear_trend_to_DC, old, extrap_to_DC, trend_to_DC, extrap_to_DC_or_zero.
# Phase methods: interp_to_DC, old, zero_DC, interp_and_shift_to_DC, trend_and_shift_to_DC.
# ZERO_PAD=True: applies Tukey window, zeros above fin[-1].
# Returns Sout = H_mag_i * exp(1j*H_ph_i), shape = len(fout).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.interp_Sparam.py_impl import interp_Sparam


def _op(debug=True):
    return SimpleNamespace(DEBUG=debug, ZERO_PAD=False)


def _param():
    p = SimpleNamespace()
    p.fb = 28e9
    p.f_r = 0.75
    return p


def _causal_sin(fin, delay=100e-12):
    """Causal (negative-phase-slope) complex signal for a transmission line."""
    return np.exp(-1j * 2 * np.pi * fin * delay)


def test_B02_D4_mag_floor_uses_machine_eps():
    """MATLAB line 8091: H_mag(H_mag<eps)=eps with eps = machine epsilon
    (2.2204e-16). Pre-fix Python floors at np.finfo(float).tiny (2.2e-308), so a
    magnitude of 1e-18 (below machine eps, above tiny) is NOT floored. After the
    fix, |Sout| is floored to ~2.2e-16.
    """
    fin = np.linspace(1e9, 20e9, 40)
    Sin = 1e-18 * _causal_sin(fin, delay=100e-12)     # |Sin| = 1e-18 everywhere
    Sout = interp_Sparam(Sin, fin, fin, 'old', 'old', _op(), _param())
    assert np.min(np.abs(Sout)) >= 0.9 * np.finfo(float).eps, (
        "min|Sout|=%g; MATLAB floors |S|<eps to machine eps 2.2204e-16 (line 8091). "
        "Pre-fix tiny-floor leaves 1e-18." % np.min(np.abs(Sout)))


def test_B02_D5_phase_extrapolates_linearly():
    """MATLAB line 8185: H_ph_i = interp1(fin,H_ph,fout,'linear','extrap') extends
    the phase slope beyond fin[-1]. Pre-fix Python clamps (left/right), freezing
    the group delay to ~0 in the extrapolated band. After the fix the group delay
    continues at the channel delay.
    """
    delay = 100e-12
    fin = np.linspace(1e9, 20e9, 40)
    fout = np.linspace(1e9, 40e9, 80)                 # extends to 2x fin[-1]
    Sin = _causal_sin(fin, delay)
    Sout = interp_Sparam(Sin, fin, fout, 'old', 'old', _op(), _param())
    ph = np.unwrap(np.angle(Sout))
    gd = -np.diff(ph) / np.diff(fout)                 # group delay
    extrap = fout[1:] > fin[-1]
    assert np.mean(gd[extrap]) > 0.5 * delay, (
        "mean group delay in the extrapolated band = %g s; expected ~%g s "
        "(linear phase extrapolation). Pre-fix clamp gives ~0." %
        (np.mean(gd[extrap]), delay))


def test_output_length_matches_fout():
    """Output has same length as fout."""
    fin = np.linspace(1e9, 20e9, 50)
    fout = np.linspace(0, 25e9, 100)
    Sin = _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'old', 'old', _op(), _param())
    assert len(Sout) == len(fout)


def test_identity_at_input_frequencies():
    """At fin points that coincide with fout, magnitude should match."""
    fin = np.linspace(1e9, 20e9, 20)
    fout = fin.copy()
    Sin = 0.8 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'old', 'old', _op(), _param())
    assert np.allclose(np.abs(Sout), np.abs(Sin), atol=1e-6)


def test_extrap_to_dc_at_zero_freq():
    """extrap_to_DC: first output at fout=0 should be finite."""
    fin = np.linspace(1e9, 20e9, 30)
    fout = np.concatenate([[0.0], fin])
    Sin = 0.7 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'extrap_to_DC', 'interp_to_DC', _op(), _param())
    assert np.isfinite(Sout[0])
    assert len(Sout) == len(fout)


def test_linear_trend_to_dc_dc_value_finite():
    """linear_trend_to_DC: DC extrapolation should be finite and positive magnitude."""
    fin = np.linspace(1e9, 20e9, 30)
    fout = np.concatenate([[0.0], fin])
    Sin = 0.7 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC', 'interp_to_DC', _op(), _param())
    assert np.isfinite(np.abs(Sout[0]))
    assert np.abs(Sout[0]) > 0


def test_anti_causal_raises_in_non_debug():
    """Non-causal response (increasing phase) raises ValueError in non-DEBUG mode."""
    fin = np.linspace(1e9, 20e9, 30)
    # positive phase slope = anti-causal
    Sin = np.exp(1j * 2 * np.pi * fin * 100e-12)
    op = _op(debug=False)
    with pytest.raises(ValueError, match='Anti-causal'):
        interp_Sparam(Sin, fin, fin, 'old', 'old', op, _param())


def test_invalid_mag_method_raises():
    """Invalid opt_interp_Sparam_mag raises ValueError."""
    fin = np.linspace(1e9, 20e9, 20)
    Sin = _causal_sin(fin)
    with pytest.raises(ValueError, match='interp_Sparam: invalid'):
        interp_Sparam(Sin, fin, fin, 'bad_method', 'old', _op(), _param())


def test_trend_to_dc_method():
    """trend_to_DC: returns array of correct length with finite values."""
    fin = np.linspace(1e9, 20e9, 25)
    fout = np.linspace(0, 20e9, 50)
    Sin = 0.6 * _causal_sin(fin)
    Sout = interp_Sparam(Sin, fin, fout, 'trend_to_DC', 'interp_to_DC', _op(), _param())
    assert len(Sout) == 50
    assert np.all(np.isfinite(Sout))


# --------------------------------------------------------------------------
# The configured default path, against COM Octave itself.
#
# Every shipped configuration workbook selects mag='linear_trend_to_DC' and
# phase='extrap_cubic_to_dc_linear_to_inf'. Until 2026-09-22 the tests above
# exercised only 'old', 'interp_to_DC', 'extrap_to_DC' and 'trend_to_DC', so
# the branch every run takes was never entered -- and all three `std` calls
# live in it. Hence SiCoPR d5bff6c, where numpy's N normalisation stood in for
# MATLAB's N-1 inside an outlier threshold.
#
# Expected values are COM Octave's own interp_Sparam, extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run on this input.
# --------------------------------------------------------------------------

_OCT_N = 6001
_OCT = {
    0: 0.99551467368075997 + 0.00032860821987716881j,    # DC: the extrapolated point
    1: 0.94673980940840263 - 0.30724845157815045j,
    3: 0.58332679250451458 - 0.80222427311898858j,
    100: 0.94951056414620771 - 0.00090894483179480043j,
    # fin ends at 40 GHz and fout runs to 60 GHz, so these three are the only
    # samples that can see the HF extrapolation. Without them the second
    # std(ddof=1), at py_impl line 249, is unverified: every index above is
    # between 10 MHz and 1 GHz, and sum|Sout| is a MAGNITUDE, so it cannot see
    # a phase method at all. COM Octave returns the identical sum|Sout| for
    # 'trend_and_shift_to_DC' and for this branch, which is the proof of that.
    # Added 2026-09-22 after tests/test_mutation_score.py reported line 249's
    # mutant surviving.
    4500: 0.55274836307238031 - 0.089594916655080884j,   # 45 GHz, extrapolated
    5000: 0.49943855721016217 - 0.16600467084272463j,    # 50 GHz, extrapolated
    6000: 0.36779627827274347 - 0.27458392270543880j,    # 60 GHz, extrapolated
}
_OCT_SUM_ABS = 4016.1085110174708


def _default_path_fixture():
    """A channel that starts at 10 MHz, so the DC-extrapolation branch runs."""
    fin = np.arange(1, 4001) * 10e6
    rng = np.random.default_rng(5)
    f_ghz = fin / 1e9
    mag = 10 ** (-(0.4 * np.sqrt(f_ghz) + 0.05 * f_ghz) / 20)
    ph = -2 * np.pi * fin * 5e-9 + 2e-3 * rng.standard_normal(fin.size)
    fmax, fstep = 60e9, fin[2] - fin[1]
    fout = np.arange(0, round(fmax / fstep) + 1) * (fmax / round(fmax / fstep))
    return mag * np.exp(1j * ph), fin, fout


def test_default_option_path_matches_com_octave():
    """mag='linear_trend_to_DC', phase='extrap_cubic_to_dc_linear_to_inf'."""
    Sin, fin, fout = _default_path_fixture()
    Sout = np.asarray(interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC',
                                    'extrap_cubic_to_dc_linear_to_inf',
                                    _op(debug=False), _param())).ravel()
    assert Sout.size == _OCT_N, 'length %d, COM Octave gives %d' % (Sout.size, _OCT_N)
    for idx, want in _OCT.items():
        rel = abs(Sout[idx] - want) / abs(want)
        assert rel < 1e-13, 'Sout[%d] is %r, COM Octave gives %r (rel %.2e)' % (
            idx, Sout[idx], want, rel)
    rel = abs(float(np.sum(np.abs(Sout))) - _OCT_SUM_ABS) / _OCT_SUM_ABS
    assert rel < 1e-13, 'sum|Sout| differs from COM Octave by %.2e' % rel


def test_default_path_is_sensitive_to_std_normalisation():
    """Guard the guard: if this input stopped straddling the outlier threshold,
    the comparison above would pass under either std convention."""
    Sin, fin, _ = _default_path_fixture()
    gd = -np.diff(np.unwrap(np.angle(Sin))) / np.diff(fin)
    lf = gd[:50]
    m = np.median(lf)
    kept = [np.abs(lf - m) < np.std(lf, ddof=d) for d in (0, 1)]
    assert not np.array_equal(kept[0], kept[1]), (
        'fixture no longer distinguishes N from N-1 normalisation; re-craft it')


# --------------------------------------------------------------------------
# phase='trend_and_shift_to_DC', against COM Octave.
#
# This option had NO test at all: it was pinned in
# tests/test_option_coverage.py's KNOWN_UNCOVERED, so the whole branch was dead
# to the suite. It carries its own std(ddof=1) at py_impl line 205 and a
# load-bearing H_ph.copy() at line 209, and tests/test_mutation_score.py
# reported both mutants surviving.
#
# Same fixture as the default path, so the two differ only in the phase method.
# Note sum|Sout| is IDENTICAL to the default branch's, because it is a
# magnitude: the HF samples are what separate them.
# --------------------------------------------------------------------------

_OCT_TS = {
    0: 0.99551472791570110 + 0.0j,                       # DC: no shift residue
    1: 0.94663833857097720 - 0.30756094300708814j,
    3: 0.58306195551151700 - 0.80241677903068590j,
    100: 0.94951021238507070 - 0.0012223675448812150j,
    4500: 0.55995000853068750 - 0.0037403096275264464j,  # 45 GHz, extrapolated
    5000: 0.52626312163823300 - 0.0065992403617536576j,  # 50 GHz, extrapolated
    6000: 0.45885344702624004 - 0.011133151922310274j,   # 60 GHz, extrapolated
}


def test_trend_and_shift_to_DC_matches_com_octave():
    """mag='linear_trend_to_DC', phase='trend_and_shift_to_DC'."""
    Sin, fin, fout = _default_path_fixture()
    Sout = np.asarray(interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC',
                                    'trend_and_shift_to_DC',
                                    _op(debug=False), _param())).ravel()
    assert Sout.size == _OCT_N, 'length %d, COM Octave gives %d' % (
        Sout.size, _OCT_N)
    for idx, want in _OCT_TS.items():
        rel = abs(Sout[idx] - want) / abs(want)
        assert rel < 1e-13, (
            'Sout[%d] is %r, COM Octave gives %r (rel %.2e)'
            % (idx, Sout[idx], want, rel))


def test_trend_and_shift_differs_from_the_default_phase_method():
    """Guard the guard: the two phase methods must actually disagree here.

    If this fixture ever stopped separating them, the oracle comparison above
    would pass while running the wrong branch, and sum|Sout| would not notice
    because it is a magnitude.
    """
    Sin, fin, fout = _default_path_fixture()
    a = np.asarray(interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC',
                                 'trend_and_shift_to_DC',
                                 _op(debug=False), _param())).ravel()
    b = np.asarray(interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC',
                                 'extrap_cubic_to_dc_linear_to_inf',
                                 _op(debug=False), _param())).ravel()
    assert abs(a[6000] - b[6000]) > 1e-3, (
        'the two phase methods now agree at 60 GHz, so this fixture no longer '
        'exercises the branch it claims to; re-craft it')
    assert abs(float(np.sum(np.abs(a))) - float(np.sum(np.abs(b)))) < 1e-9, (
        'sum|Sout| now differs between phase methods; the comment above, and '
        'the reason the HF samples exist, need revisiting')


def test_hf_extrapolation_is_sensitive_to_std_normalisation():
    """Guard the guard, for the SECOND std: the HF outlier mask at py_impl
    line 249 must straddle N versus N-1 on this fixture, or the HF samples
    pinned above would pass under either convention."""
    Sin, fin, _ = _default_path_fixture()
    gd = -np.diff(np.unwrap(np.angle(Sin))) / np.diff(fin)
    hf = gd[-51:]                      # MATLAB group_delay(end-50:end) is 51
    m = np.median(hf)
    kept = [np.abs(hf - m) < np.std(hf, ddof=d) for d in (0, 1)]
    assert not np.array_equal(kept[0], kept[1]), (
        'fixture no longer distinguishes N from N-1 in the HF trend; '
        're-craft it')


# --------------------------------------------------------------------------
# The remaining option branches, against COM Octave.
#
# All four were in test_option_coverage.py's KNOWN_UNCOVERED, so no test
# selected them and the branches were dead to the suite. Sweeping them against
# the reference found two real defects at once, both the same class: MATLAB's
# interp1 is called with 'linear','extrap' at EVERY site in this function, and
# the port used bare np.interp, which CLAMPS.
#
#   extrap_to_DC_or_zero    clamped at DC, giving H_mag[0] instead of the
#                           extrapolated value. 0.995348 against 0.997306.
#   interp_and_shift_to_DC  clamped above fin[-1]=40 GHz, so the whole
#                           extrapolated band was wrong by up to 198%.
#
# Ten sites were converted to the file's own _interp_extrap helper. All four
# options now agree with COM Octave to 1e-15.
# --------------------------------------------------------------------------

_OCT_EXTRAP_OR_ZERO = {
    0: 0.99730593480744656 + 0.00032919949506952662j,   # DC, extrapolated
    1: 0.94673980940840263 - 0.30724845157815045j,
    3: 0.58332679250451458 - 0.80222427311898858j,
    100: 0.94951056414620771 - 0.00090894483179480043j,
    4500: 0.58597274243719866 - 0.094980252368355333j,
    5000: 0.56331832643948188 - 0.18723719266414290j,
    6000: 0.47567949416826988 - 0.35512578341644940j,
}

_OCT_INTERP_AND_SHIFT = {
    0: 0.99551472791570106 + 0.0j,
    1: 0.94631042174646207 - 0.30856840961100102j,
    3: 0.58306478459292188 - 0.80241472332111674j,
    100: 0.94951092389725633 - 0.00037816099156726970j,
    4500: 0.081019575306351607 - 0.55407023957870660j,  # extrapolated band
    5000: -0.50427908000090371 - 0.15066198141273707j,
    6000: 0.38378038150916721 + 0.25175990881843324j,
}

_OCT_ZERO_DC = {
    0: 0.99551472791570106 + 0.0j,
    1: 0.94613778177939767 - 0.30909735619281908j,
    3: 0.58261613749868002 - 0.80274053564088432j,
    100: 0.94951056414620771 - 0.00090894483179480043j,
    4500: 0.080709833127870334 - 0.55411544357299602j,
    5000: -0.50436322241472187 - 0.15038006199362844j,
    6000: 0.38392105726820769 + 0.25154533330127454j,
}


def _check_against(mag_method, ph_method, expected):
    Sin, fin, fout = _default_path_fixture()
    Sout = np.asarray(interp_Sparam(Sin, fin, fout, mag_method, ph_method,
                                    _op(debug=False), _param())).ravel()
    assert Sout.size == _OCT_N
    for idx, want in expected.items():
        rel = abs(Sout[idx] - want) / abs(want)
        assert rel < 1e-13, (
            'mag=%s ph=%s Sout[%d] is %r, COM Octave gives %r (rel %.2e)'
            % (mag_method, ph_method, idx, Sout[idx], want, rel))


def test_mag_extrap_to_DC_or_zero_matches_com_octave():
    """Was clamping at DC where ML 8496 extrapolates log10(H_mag)."""
    _check_against('extrap_to_DC_or_zero', 'extrap_cubic_to_dc_linear_to_inf',
                   _OCT_EXTRAP_OR_ZERO)


def test_phase_interp_and_shift_to_DC_matches_com_octave():
    """Was clamping above fin[-1] where ML 8559 extrapolates."""
    _check_against('linear_trend_to_DC', 'interp_and_shift_to_DC',
                   _OCT_INTERP_AND_SHIFT)


def test_phase_zero_DC_matches_com_octave():
    _check_against('linear_trend_to_DC', 'zero_DC', _OCT_ZERO_DC)


def test_linear_trend_to_DC_log_trend_to_inf_is_refused_like_the_reference():
    """UPSTREAM: the reference cannot run this option on an ordinary channel.

    hf_logtrend_val is assigned only inside `if hf_trend_val>H_mag(end)` or
    `elseif hf_trend_val<eps`. For a normal HF trend, between the two, neither
    branch runs and ML 8494 reads an undefined variable.

        COM Octave: error: 'hf_logtrend_val' undefined
                    interp_Sparam at line 55 column 13

    MATLAB raises the same. The port used to pre-initialise the variable to
    H_mag[-1] and answer where the reference stops; it now refuses, and names
    the upstream defect in the message.
    """
    Sin, fin, fout = _default_path_fixture()
    with pytest.raises(ValueError, match='hf_logtrend_val'):
        interp_Sparam(Sin, fin, fout, 'linear_trend_to_DC_log_trend_to_inf',
                      'extrap_cubic_to_dc_linear_to_inf',
                      _op(debug=False), _param())
