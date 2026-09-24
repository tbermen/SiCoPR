"""Integration tests for get_ILN_cmp_td (MATLAB lines ~6270).

Computes the time-domain insertion-loss noise (TD ILN): fits a smooth complex
model to sdd21, builds REF (raw) and FIT (fitted) pulse responses through the
TX/Bessel-Thomson filters, and reports a FOM from their difference PDF.

Bessel_Thomson_Filter / Butterworth_Filter / s21_to_impulse_DC are top-level fns
in the assembled sicopr.py, so this is an INTEGRATION test: the real dependencies
are injected from `sicopr`. Both the zero-input (degenerate) and a realistic lossy
channel are exercised.
"""
import os
import sys
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sicopr
import com_functions.fn.get_ILN_cmp_td.py_impl as _mod
from com_functions.fn.get_ILN_cmp_td.py_impl import get_ILN_cmp_td

for _n in dir(sicopr):
    _v = getattr(sicopr, _n)
    if callable(_v) and not _n.startswith('__') and not hasattr(_mod, _n):
        setattr(_mod, _n, _v)


def _op():
    return SimpleNamespace(
        BinSize=1e-4,
        transmitter_transition_time=10e-12,
        impulse_response_truncation_threshold=1e-7,
        DEBUG=True,  # downgrade the anti-causal check to a warning for synthetic channels
        ENFORCE_CAUSALITY=0,
        EC_PULSE_TOL=0.01, EC_REL_TOL=1e-2, EC_DIFF_TOL=1e-3,
    )


def _param(fb=26.5625e9, spui=4):
    p = SimpleNamespace()
    p.samples_per_ui = spui
    p.samples_for_C2M = 8
    p.sample_dt = 1.0 / (fb * spui)
    p.specBER = 1e-4
    p.levels = 4
    p.fb = fb
    p.BTorder = 4
    p.fb_BT_cutoff = 0.75
    p.fb_BW_cutoff = 0.75
    return p


def _channel(fb=26.5625e9, n=200, tau=200e-12, loss_db_per_ghz=0.6):
    f = np.linspace(0, 2 * fb, n)
    mag = np.exp(-(loss_db_per_ghz / 20.0) * np.log(10) * f / 1e9)
    sdd21 = mag * np.exp(-1j * 2 * np.pi * f * tau)
    return sdd21, f


_TD_FIELDS = ('FOM', 'FOM_PDF', 'SNR_ISI_FOM', 'SNR_ISI_FOM_PDF', 'REF', 'FIT')


def test_zero_sdd21_is_refused_like_the_reference():
    """An all-zero channel has no answer, and the reference does not invent one.

    These tests used to assert that an all-zero sdd21 returned arrays of the
    input length with every TD field present. It did, but only because the port
    floored its logarithms with machine epsilon: log(0) became log(eps) and the
    fit matrix became invertible. Both floors are absent from ML 6731 and
    ML 6738 and have been removed, so the question is what the reference does,
    and the answer is that it stops:

        COM Octave, sdd21 = zeros(40,1):
            warning: inverse: matrix singular to machine precision, rcond = 0
            error: IL(0): subscripts must be either integers 1 to (2^63)-1
                s21_to_impulse_DC at line 23 column 9
                get_ILN_cmp_td at line 52 column 6

    SiCoPR stops at the singular inverse itself, a few lines before the
    reference stops on the subscript, but neither returns a result. Asserting
    the refusal is the honest version; asserting shapes was asserting behaviour
    that only the epsilon floor produced.
    """
    N = 50
    f = np.linspace(0, 26.5625e9, N)
    with pytest.raises(np.linalg.LinAlgError, match='[Ss]ingular'):
        get_ILN_cmp_td(np.zeros(N, dtype=complex), f, _op(), _param())


def test_nonzero_channel_computes_iln():
    """Realistic lossy channel: ILN/efit have input length and TD has all fields."""
    sdd21, f = _channel()
    ILN, efit, TD = get_ILN_cmp_td(sdd21, f, _op(), _param())
    assert len(ILN) == len(f)
    assert len(efit) == len(f)
    assert np.all(np.isfinite(efit))
    for field in _TD_FIELDS:
        assert hasattr(TD, field), f'missing TD field: {field}'
    assert np.isfinite(float(TD.FOM))


def test_iln_shape_matches_input():
    """Shape on a channel that actually has an answer.

    This used an all-zero sdd21, which the reference cannot run at all; see
    test_zero_sdd21_is_refused_like_the_reference. The shape claim is worth
    keeping, so it is made on a real channel instead, and it now also checks
    the TD fields the deleted test covered.
    """
    sdd21, f = _channel()
    ILN, efit, TD = get_ILN_cmp_td(sdd21, f, _op(), _param())
    assert len(ILN) == len(f)
    assert len(efit) == len(f)
    for field in _TD_FIELDS:
        assert hasattr(TD, field), 'missing TD field: %s' % field


def test_uses_real_s21_to_impulse_DC():
    assert _mod.s21_to_impulse_DC is sicopr.s21_to_impulse_DC
    assert _mod.Bessel_Thomson_Filter is sicopr.Bessel_Thomson_Filter


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# --------------------------------------------------------------------------
# Against COM Octave itself.
#
# Every test above pins values the PORT produced, which cannot catch a
# divergence that lives in a library default. Driving the reference found two
# defect classes in this function.
#
# 1. THE FIT. ML 6740 is
#        alpha = ((fmbg'*fmbg)^-1)*fmbg'*LGw
#    the normal equations with an EXPLICIT INVERSE. The port used
#    np.linalg.lstsq. Those agree only for a well-conditioned system, and this
#    one is not: ML 6737 is warning('off','MATLAB:nearlySingularMatrix'), so
#    the reference knows and proceeds. lstsq truncates the small singular
#    values and returns the minimum-norm answer; the inverse amplifies them.
#    The two fits differed by 0.9 dB across the band. Measured on a smooth
#    sdd21 = 0.9*exp(-f/40e9), where the 4-term fit is near-exact:
#        COM Octave ILN[0:3] = 3.33e-14, 5.33e-14, 6.06e-14
#        lstsq      ILN[0:3] = -0.905,   -0.807,   -0.713
#    The reference reproduces the channel essentially exactly, which is what
#    ILN is for. The lstsq fit did not.
#
# 2. TWO EPSILON FLOORS. ML 6731 is db = @(x) 20*log10(abs(x)) and ML 6738 is
#    log(abs(sdd21)), both with NO floor. The port added
#    np.finfo(float).eps inside each. Same defect plot_modal carried: at
#    |x| = 0 the reference gives -Inf and the floored version gives a
#    plausible -313 dB that reads like a measurement.
#
# The fixture carries RIPPLE on purpose. A smooth exponential is reproduced
# almost exactly by the 4-term fit, which drives TD_ILN.FOM to ~1e-15 and makes
# SNR_ISI_FOM a ratio of noise-floor numbers that cannot discriminate anything.
# The ripple is the thing ILN exists to measure.
#
# Real-valued sdd21 on purpose too: Octave's abs() of a COMPLEX number and
# numpy's differ by an ULP, which would force a tolerance wide enough to hide
# the epsilon floors.
#
# Tolerance is 1e-11 absolute, not tighter: both sides invert a nearly singular
# matrix, so the last digits follow the inversion path. That is the honest
# floor for this computation, and it is still 11 orders below the 0.9 dB defect.
# --------------------------------------------------------------------------

_OCT_N_ILN = 40


def _oct_channel():
    f = np.linspace(0.05e9, 20e9, _OCT_N_ILN)
    sdd21 = 0.9 * np.exp(-f / 40e9) * (1.0 + 0.05 * np.cos(f / 1.3e9))
    return sdd21, f


def _oct_param():
    return SimpleNamespace(
        fb=25e9, sample_dt=1e-12, f_r=0.75, fb_BT_cutoff=0.473, Z0=50.0,
        samples_per_ui=8, ndfe=4, levels=4, P_peak=1e-4, T_r=8e-3,
        BTorder=4, Butterworth_order=4, f_bt=0.75, f_bw=0.75,
        fb_BW_cutoff=0.75, BWorder=4, BinSize=1e-5, sigma_X=0.5, R_LM=1.0,
        eta_0=5.2e-8, A_v=0.413, A_fe=0.413, A_ne=0.608, SNR_TX=32,
        Qbit_max=6, specBER=1e-5, f1=0.05e9, f2=20e9, Grr=1, Grr_limit=1)


def _oct_op():
    return SimpleNamespace(
        DEBUG=False, DISPLAY_WINDOW=False,
        interp_sparam_mag='linear_trend_to_DC',
        interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf',
        ZERO_PAD=False, ENFORCE_CAUSALITY=False,
        EC_PULSE_TOL=0.05, EC_REL_TOL=1e-3, EC_DIFF_TOL=1e-5,
        impulse_response_truncation_threshold=1e-3, TDMODE=0,
        Bessel_Thomson=False, Butterworth=False,
        transmitter_transition_time=8e-3, BinSize=1e-5, nburst=1, T_O=0,
        RX_CALIBRATION=False, include_pdf_extras=False)


_OCT_ILN_HEAD = [-0.15540757104715852, 0.18306834436229258,
                 0.22424211093070623]
_OCT_EFIT_HEAD = [-0.34711951585160572, -0.83444244927628686,
                  -1.0827399119718928]
_OCT_TD_ILN = {
    'FOM': 0.012639707535847632,
    'FOM_PDF': 0.037280000000000008,
    'SNR_ISI_FOM': 27.857231811948793,
    'SNR_ISI_FOM_PDF': 18.462454240912891,
}


def test_octave_fit_uses_the_normal_equations():
    """ILN and efit against COM Octave. Fails by ~0.9 dB under lstsq."""
    sdd21, f = _oct_channel()
    ILN, efit, _TD = get_ILN_cmp_td(sdd21, f, _oct_op(), _oct_param(), 1.0)
    ILN = np.ravel(np.asarray(ILN)).astype(float)
    efit = np.ravel(np.asarray(efit)).astype(float)
    assert ILN.size == _OCT_N_ILN and efit.size == _OCT_N_ILN
    np.testing.assert_allclose(ILN[:3], _OCT_ILN_HEAD, rtol=0, atol=1e-11)
    np.testing.assert_allclose(efit[:3], _OCT_EFIT_HEAD, rtol=0, atol=1e-11)


def test_octave_td_iln_scalars():
    """The four TD_ILN figures of merit, against COM Octave."""
    sdd21, f = _oct_channel()
    _ILN, _efit, TD = get_ILN_cmp_td(sdd21, f, _oct_op(), _oct_param(), 1.0)
    for name, want in sorted(_OCT_TD_ILN.items()):
        got = float(np.ravel(np.asarray(getattr(TD, name)))[0])
        assert abs(got - want) <= 1e-9 * max(abs(want), 1e-300), (
            'TD_ILN.%s is %.17g, COM Octave gives %.17g' % (name, got, want))


def test_octave_fixture_still_has_a_fit_residual():
    """Guard the guard: a channel the 4-term fit reproduces exactly drives
    TD_ILN.FOM to ~1e-15, and then SNR_ISI_FOM is a ratio of noise-floor
    numbers that agrees under any fit. The ripple must keep FOM meaningful."""
    sdd21, f = _oct_channel()
    _ILN, _efit, TD = get_ILN_cmp_td(sdd21, f, _oct_op(), _oct_param(), 1.0)
    assert float(np.ravel(np.asarray(TD.FOM))[0]) > 1e-4, (
        'fixture no longer leaves a fit residual; re-craft the ripple')


# --------------------------------------------------------------------------
# COM Octave, 4p15p0: the same fixture carried through a 150 ps delay,
# sdd21 * exp(-2j*pi*f*150e-12), so fmbg is COMPLEX. ML 6740 uses `'`, the
# conjugate transpose; every fixture above is real, where `'` and `.'` agree
# and no test could tell fmbg.conj().T from fmbg.T. On this one the plain
# transpose moves ILN by 2.57 while the port agrees with Octave to 1.4e-12,
# inside the 1e-11 floor argued above.
# --------------------------------------------------------------------------

_OCT_C_ILN_HEAD = [-0.15540757104715758, 0.18306834436232755,
                   0.22424211093077839]
_OCT_C_EFIT_HEAD = [-0.34711951585160666, -0.83444244927632294,
                    -1.0827399119719661]


def test_octave_fit_on_complex_channel_uses_the_conjugate_transpose():
    sdd21, f = _oct_channel()
    sdd21 = sdd21 * np.exp(-2j * np.pi * f * 150e-12)
    ILN, efit, _TD = get_ILN_cmp_td(sdd21, f, _oct_op(), _oct_param(), 1.0)
    ILN = np.ravel(np.asarray(ILN)).astype(float)
    efit = np.ravel(np.asarray(efit)).astype(float)
    np.testing.assert_allclose(ILN[:3], _OCT_C_ILN_HEAD, rtol=0, atol=1e-11)
    np.testing.assert_allclose(efit[:3], _OCT_C_EFIT_HEAD, rtol=0, atol=1e-11)
