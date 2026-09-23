"""Verification tests for SNDR_ref().

# ============================================================
# MATLAB GROUND TRUTH (lines 4405-4442)
# For each of 6 TX-FFE presets, apply FFE and compute:
#   SNDR = 10*log10(ss(hss) * 10^(SNR_TX/10) / ss(PR_noFFE_sampled))
# Default presets: 6 standard PAM4 TX-FFE configurations.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.SNDR_ref.py_impl import SNDR_ref


def _param(M=8, D_p=2, N_p=20):
    return SimpleNamespace(
        SNDR=[20.0],
        samples_per_ui=M,
        D_p=D_p,
        N_p=N_p,
    )


def _gaussian_pr(N=400, peak=100, M=8):
    # N was 200 and peak 50, which puts N_p*M+ipeak at 211 past a 200-sample
    # pulse.  COM Octave refuses that: "error: PR_FFE(211): out of bound 200
    # (dimensions are 200x1)".  These tests were exercising a call the
    # reference declines; 400/100 keeps their intent inside its range.
    x = np.arange(N)
    return np.exp(-0.5 * ((x - peak) / 5) ** 2)


def test_returns_six_presets():
    """SNDR_ref has 6 values (one per preset)."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    assert len(results.SNDR_ref) == 6


def test_sndr_values_are_finite():
    """All SNDR_ref values are finite."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    assert np.all(np.isfinite(results.SNDR_ref))


def test_preset_fields_present():
    """Results has all per-preset SNDR fields."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    for field in ('SNDR_ref_p1', 'SNDR_ref_p2', 'SNDR_ref_p3',
                  'SNDR_ref_p4', 'SNDR_ref_p5', 'SNDR_ref_p6'):
        assert hasattr(results, field)


def test_sigma_il_is_positive():
    """sigma_iL (first preset) is positive."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    assert results.sigma_iL > 0


def test_custom_presets_used():
    """When param.preset is provided with 6 entries, those are used."""
    param = _param()
    param.preset = [SimpleNamespace(txffe=[0, 0, 0, 1, 0])] * 6
    PR = _gaussian_pr()
    results = SNDR_ref(PR, param)
    assert len(results.SNDR_ref) == 6


# ---------------------------------------------------------------------------
# COM Octave oracle tests.
#
# Values below come from running SNDR_ref verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m through tools/octave_oracle.py.
# Two deviations from a plain oracle run, both recorded here because they
# change what the oracle is:
#
#  * FFE is taken from matlab/com_ieee8023_4p16p0.m, not from the compat file,
#    because the compat FFE is a rewritten speed variant.
#  * The compat SNDR_ref body spells MATLAB's abbreviated
#    find(PR_FFE==max(PR_FFE),1,'fir') which Octave rejects with
#    'find: DIRECTION must be "first" or "last"'.  MATLAB accepts any
#    unambiguous prefix, so the probe ran it with 'first' spelled out.  That
#    makes SNDR_ref unrunnable under Octave as the compat file stands.
#
# Tolerance is 1e-13: sum() accumulates in a different order in each of the
# three languages, and the measured worst case over these probes is 8e-16.
# ---------------------------------------------------------------------------

_OCT_M, _OCT_DP, _OCT_NP = 8, 2, 20


def _oct_pr(N=400, peak=100, w=5.0):
    x = np.arange(N, dtype=float)
    return np.exp(-0.5 * ((x - peak) / w) ** 2)


def _oct_param(**over):
    p = SimpleNamespace(SNDR=[20.0], samples_per_ui=_OCT_M, D_p=_OCT_DP,
                        N_p=_OCT_NP)
    for k, v in over.items():
        setattr(p, k, v)
    return p


def test_oracle_nominal_values():
    """The whole result struct, pinned.  This function had no value pin."""
    r = SNDR_ref(_oct_pr(), _oct_param())
    np.testing.assert_allclose(
        r.SNDR_ref,
        [19.999999999629168, 13.979400086349544, 17.106444398674082,
         16.495872304288696, 14.968160618508064, 17.50122526746317],
        rtol=1e-13)
    assert r.SNDR_ref_p1 == pytest.approx(19.999999999629168, rel=1e-13)
    assert r.SNDR_ref_p2 == pytest.approx(13.979400086349544, rel=1e-13)
    assert r.SNDR_ref_p3 == pytest.approx(17.106444398674082, rel=1e-13)
    assert r.SNDR_ref_p4 == pytest.approx(16.495872304288696, rel=1e-13)
    assert r.SNDR_ref_p5 == pytest.approx(14.968160618508064, rel=1e-13)
    assert r.SNDR_ref_p6 == pytest.approx(17.50122526746317, rel=1e-13)
    assert r.sigma_iL == pytest.approx(1.0745607971094413, rel=1e-13)


def test_oracle_m_32_and_m_1():
    """samples_per_ui only enters through the subsampling stride."""
    r = SNDR_ref(_oct_pr(N=1600, peak=400, w=20.0),
                 _oct_param(samples_per_ui=32))
    np.testing.assert_allclose(
        r.SNDR_ref,
        [19.999999999629168, 13.979400086349544, 17.102301477455462,
         16.562663047453718, 15.016421367035743, 17.50122526746317],
        rtol=1e-13)
    r1 = SNDR_ref(_oct_pr(N=64, peak=16, w=3.0), _oct_param(samples_per_ui=1))
    np.testing.assert_allclose(
        r1.SNDR_ref,
        [19.459116987544405, 13.438517074264784, 16.129649029139653,
         15.112019843562363, 12.815808638253326, 16.960342255378411],
        rtol=1e-13)
    assert r1.sigma_iL == pytest.approx(2.1667257513590861, rel=1e-13)


def test_oracle_negative_n_p_gives_a_short_hss():
    """N_p<0 makes a still-legal, shorter range -- MATLAB does not object."""
    r = SNDR_ref(_oct_pr(), _oct_param(N_p=-1))
    np.testing.assert_allclose(
        r.SNDR_ref,
        [8.2594472854683953, 2.2388473721887712, 1.9516194366030135,
         -0.89174684909265411, -13.529968063642666, 5.7606725533023964],
        rtol=1e-13)


def test_oracle_fractional_d_p_is_not_truncated():
    """D_p=2.5 still makes a whole subscript, so MATLAB answers -- differently.

    COM Octave with D_p=2.5: SNDR_ref(1)=19.632072643603557 and
    sigma_iL=1.0299938331627811.  int(param.D_p) moved the start of hss by two
    samples and gave 19.999999999629168 / 1.0745607971094413 instead.
    """
    r = SNDR_ref(_oct_pr(), _oct_param(D_p=2.5))
    np.testing.assert_allclose(
        r.SNDR_ref,
        [19.632072643603557, 13.611472730323932, 16.652585356112549,
         16.039090340708377, 14.377435359573923, 17.133297911437559],
        rtol=1e-13)
    assert r.sigma_iL == pytest.approx(1.0299938331627811, rel=1e-13)


def test_oracle_peak_skips_nan_where_argmax_does_not():
    """MATLAB max() ignores NaN; np.argmax returns the NaN's index.

    COM Octave on a 400-point Gaussian with PR(101)=NaN (the sample at the
    peak): SNDR_ref(1)=19.999999999947136 and sigma_iL=1.0681501326056455.
    np.argmax anchored on the NaN and made every output NaN.
    """
    pr = _oct_pr()
    pr[100] = np.nan
    r = SNDR_ref(pr, _oct_param())
    np.testing.assert_allclose(
        r.SNDR_ref,
        [19.999999999947136, 13.979400086667512, 17.094838858435825,
         16.547846175422023, 15.020134489641395, 17.501225267781138],
        rtol=1e-13)
    assert r.sigma_iL == pytest.approx(1.0681501326056455, rel=1e-13)


def test_oracle_nan_away_from_the_peak_only_poisons_sigma_tn():
    """A NaN off the peak leaves the peak, and sigma_iL, alone.

    COM Octave with PR(301)=NaN: SNDR_ref all NaN but
    sigma_iL=1.0745607971094413, the nominal value.
    """
    pr = _oct_pr()
    pr[300] = np.nan
    r = SNDR_ref(pr, _oct_param())
    assert np.all(np.isnan(r.SNDR_ref))
    assert r.sigma_iL == pytest.approx(1.0745607971094413, rel=1e-13)


def test_oracle_all_nan_pulse_matches_nothing():
    """max() of an all-NaN vector is NaN, which == matches nothing, so MATLAB
    indexes with an empty subscript: SNDR_ref all NaN, sigma_iL exactly 0."""
    r = SNDR_ref(np.full(400, np.nan), _oct_param())
    assert np.all(np.isnan(r.SNDR_ref))
    assert r.sigma_iL == 0.0


def test_oracle_hss_start_before_the_array_is_an_error():
    """COM Octave, peak 10 samples in with D_p*M=16:
    "error: PR_FFE(-5): subscripts must be either integers 1 to (2^63)-1 or
    logicals".  A Python slice with a negative start read from the tail
    instead and returned -inf for every preset.
    """
    with pytest.raises(IndexError, match=r'PR_FFE\(-5\)'):
        SNDR_ref(_oct_pr(N=400, peak=10), _oct_param())
    # all zeros: ipeak is 1, so the start is -15
    with pytest.raises(IndexError, match=r'PR_FFE\(-15\)'):
        SNDR_ref(np.zeros(400), _oct_param())


def test_oracle_hss_end_past_the_array_is_an_error():
    """COM Octave, peak at sample 381 with N_p*M=160:
    "error: PR_FFE(541): out of bound 400 (dimensions are 400x1)".

    This is the insidious one: the truncating slice returned
    17.106444447519134 for preset 3 against the nominal 17.106444398674082,
    close enough to pass for a rounding difference.
    """
    with pytest.raises(IndexError, match=r'PR_FFE\(541\).*out of bound 400'):
        SNDR_ref(_oct_pr(N=400, peak=380), _oct_param())


def test_oracle_non_integer_subscript_is_refused():
    """D_p=2.1 makes the hss start 84.2, which MATLAB refuses as a subscript.

    Octave accepts a non-integer RANGE as an index with
    "warning: non-integer range used as index" and truncates toward zero,
    giving SNDR_ref(1)=19.948026128813808; MATLAB errors.  Octave's answer is
    not pinned here because it is an Octave backwards-compatibility extension.
    What matters for the port is that int(param.D_p) matched NEITHER: it
    returned 19.999999999629168, the D_p=2 answer.
    """
    with pytest.raises(ValueError, match=r'PR_FFE\(84.2\)'):
        SNDR_ref(_oct_pr(), _oct_param(D_p=2.1))
    # A fractional samples_per_ui cannot index either.  COM Octave reaches
    # FFE first: "error: circshift: all values of N must be integers".
    with pytest.raises(ValueError, match='samples_per_ui'):
        SNDR_ref(_oct_pr(), _oct_param(samples_per_ui=8.5))


def test_oracle_present_but_empty_preset_is_not_defaulted():
    """`~isfield(param,'preset')` tests existence, not emptiness.

    COM Octave with param.preset=[]: the loop never runs, the local SNDR_ref
    is never assigned, so `results.SNDR_ref=SNDR_ref` resolves to the function
    and recurses -- "error: 'param' undefined near line 2, column 13".
    Substituting the six default presets answered where MATLAB refuses.
    """
    with pytest.raises(IndexError):
        SNDR_ref(_oct_pr(), _oct_param(preset=[]))


def test_oracle_eight_presets_all_reported():
    """More than six presets is fine; SNDR_ref carries them all."""
    p = _oct_param(preset=[SimpleNamespace(txffe=[0, 0, 0, 1 - i * 0.1, 0])
                           for i in range(8)])
    r = SNDR_ref(_oct_pr(), p)
    np.testing.assert_allclose(
        r.SNDR_ref,
        [19.999999999629168, 19.084850188415665, 18.061799739468039,
         16.901960799914303, 15.563025007302041, 13.979400086349544,
         12.041199826188416, 9.5424250940224162],
        rtol=1e-13)


def test_oracle_sndr_is_read_by_linear_index():
    """param.SNDR(1) on a 2x2 is element (1,1); a negative SNDR just shifts."""
    r = SNDR_ref(_oct_pr(),
                 _oct_param(SNDR=np.array([[20.0, 30.0], [40.0, 50.0]])))
    assert r.SNDR_ref_p1 == pytest.approx(19.999999999629168, rel=1e-13)
    rn = SNDR_ref(_oct_pr(), _oct_param(SNDR=[-20.0]))
    np.testing.assert_allclose(
        rn.SNDR_ref,
        [-20.000000000370832, -26.020599913650457, -22.893555601325918,
         -23.504127695711304, -25.031839381491938, -22.49877473253683],
        rtol=1e-13)
