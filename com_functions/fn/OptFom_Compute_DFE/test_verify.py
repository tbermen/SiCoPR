"""Verification tests for OptFom_Compute_DFE().

# ============================================================
# MATLAB GROUND TRUTH (lines 3219-3304)
# Returns (THIS, param) with dfetaps, excess_dfe_cursors, tail_RSS.
# cursor_i is 0-based. dfecursors at cursor_i+M:M:cursor_i+ndfe*M+1.
# Floating DFE path off; do_C2M=False tested here.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Compute_DFE.py_impl import OptFom_Compute_DFE


def _param(M=8, ndfe=3):
    bmax = np.ones(ndfe) * 0.5
    bmin = -bmax
    return SimpleNamespace(
        samples_per_ui=M,
        ndfe=ndfe,
        dfe_delta=0,
        Floating_DFE=False,
        use_bmax=bmax,
        use_bmin=bmin,
        bmax=bmax,
        bmin=bmin,
        N_tail_start=0,
        B_float_RSS_MAX=0.5,
    )


def _THIS(cursor_i=40):
    return SimpleNamespace(cursor_i=cursor_i)


def _sbr(cursor_i=40, M=8, ndfe=3, N=200):
    sbr = np.zeros(N)
    sbr[cursor_i] = 1.0
    for n in range(1, ndfe + 1):
        sbr[cursor_i + n * M] = 0.3 * (0.5 ** n)
    return sbr


def test_returns_two_outputs():
    """OptFom_Compute_DFE returns (THIS, param)."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert hasattr(THIS, 'dfetaps')


def test_dfetaps_length():
    """dfetaps has length == ndfe."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(ndfe=3), False, 0)
    assert len(THIS.dfetaps) == 3


def test_dfetaps_within_bmax():
    """dfetaps values are clipped to bmax."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert np.all(np.asarray(THIS.dfetaps) <= 0.5 + 1e-12)
    assert np.all(np.asarray(THIS.dfetaps) >= -0.5 - 1e-12)


def test_excess_dfe_cursors_shape():
    """excess_dfe_cursors has same length as ndfe."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(ndfe=3), False, 0)
    assert len(THIS.excess_dfe_cursors) == 3


def test_tail_rss_zero_when_N_tail_start_zero():
    """tail_RSS=0 when N_tail_start=0 (no tail computation)."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert THIS.tail_RSS == 0.0


def test_floating_tap_coef_empty_no_floating():
    """floating_tap_coef is empty when Floating_DFE=False."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert len(THIS.floating_tap_coef) == 0


# ---------------------------------------------------------------------------
# COM Octave oracle tests.
#
# Values below come from running OptFom_Compute_DFE verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m (with dfe_clipper, floatingDFE and
# findbankloc) through tools/octave_oracle.py.  The extracted body was first
# checked identical to matlab/com_ieee8023_4p16p0.m.  THIS.cursor_i is passed
# to Octave as cursor+1 because the port keeps it 0-based.
#
# The pinned arrays are bit-exact against the reference, so they are compared
# with rtol=0 unless noted.
# ---------------------------------------------------------------------------

_OCT_M, _OCT_CUR = 8, 40


def _oct_sbr(N=300, cursor=_OCT_CUR, taps=(0.30, -0.18, 0.09, 0.04, -0.02),
             pre=0.12):
    s = np.zeros(N)
    s[cursor] = 1.0
    if cursor - _OCT_M >= 0:
        s[cursor - _OCT_M] = pre
    for n, v in enumerate(taps, start=1):
        if cursor + n * _OCT_M < N:
            s[cursor + n * _OCT_M] = v
    for k in range(len(taps) + 1, (N - cursor) // _OCT_M):
        s[cursor + k * _OCT_M] = 0.05 * (-0.8) ** k
    return s


def _oct_param(ndfe=5, bmax=None, bmin=None, **over):
    bmax = np.full(ndfe, 0.5) if bmax is None else np.asarray(bmax, float)
    bmin = -np.full(ndfe, 0.5) if bmin is None else np.asarray(bmin, float)
    p = SimpleNamespace(samples_per_ui=_OCT_M, ndfe=ndfe, dfe_delta=0,
                        Floating_DFE=False, N_tail_start=0,
                        B_float_RSS_MAX=0.5, bmax=bmax, bmin=bmin)
    for k, v in over.items():
        setattr(p, k, v)
    return p


_OCT_FLOAT = dict(Floating_DFE=True, ndfe_passed=5, N_bf=4, N_bg=2,
                  N_bmax=20, bmaxg=0.05)


def test_oracle_tail_rss_rescale_arithmetic_order():
    """min(...)*sign(t).*t/tail_RSS: the divide comes LAST.

    COM Octave, N_tail_start=3 and B_float_RSS_MAX=0.05 on the probe pulse:
    use_bmax(3)=0.044776673559449504.  Factoring the division out into a
    scale first -- algebraically identical -- gave 0.04477667355944951.
    """
    THIS, param = OptFom_Compute_DFE(
        _oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
        _oct_param(N_tail_start=3, B_float_RSS_MAX=0.05), False, 0)
    np.testing.assert_array_equal(
        np.asarray(THIS.dfetaps),
        [0.29999999999999999, -0.17999999999999999, 0.044776673559449504,
         0.019900743804199782, -0.0099503719020998908])
    np.testing.assert_array_equal(
        np.asarray(param.use_bmax).ravel(),
        [0.5, 0.5, 0.044776673559449504, 0.019900743804199782,
         0.0099503719020998908])
    np.testing.assert_array_equal(
        np.asarray(param.use_bmin).ravel(),
        [-0.5, -0.5, -0.044776673559449504, -0.019900743804199782,
         -0.0099503719020998908])
    assert THIS.tail_RSS == 0.10049875621120891
    np.testing.assert_array_equal(
        np.asarray(THIS.excess_dfe_cursors),
        [0, 0, 0.045223326440550493, 0.020099256195800219,
         -0.01004962809790011])


def test_oracle_tail_rss_rescale_from_tap_two():
    """The same rescale with N_tail_start=2 and B_float_RSS_MAX=0.2."""
    THIS, param = OptFom_Compute_DFE(
        _oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
        _oct_param(N_tail_start=2, B_float_RSS_MAX=0.2), False, 0)
    np.testing.assert_array_equal(
        np.asarray(THIS.dfetaps),
        [0.29999999999999999, -0.17462565002615973, 0.087312825013079867,
         0.038805700005813279, -0.019402850002906639])
    assert THIS.tail_RSS == 0.20615528128088301


def test_oracle_nominal_and_dfe_delta():
    """dfe_delta quantises the cursors before clipping."""
    THIS, _ = OptFom_Compute_DFE(_oct_sbr(),
                                 SimpleNamespace(cursor_i=_OCT_CUR),
                                 _oct_param(), False, 0)
    np.testing.assert_array_equal(
        np.asarray(THIS.dfetaps),
        [0.29999999999999999, -0.17999999999999999, 0.089999999999999997,
         0.040000000000000001, -0.02])
    assert THIS.tail_RSS == 0.0
    THISq, _ = OptFom_Compute_DFE(_oct_sbr(),
                                  SimpleNamespace(cursor_i=_OCT_CUR),
                                  _oct_param(dfe_delta=0.02), False, 0)
    np.testing.assert_array_equal(
        np.asarray(THISq.dfetaps),
        [0.29999999999999999, -0.17999999999999999, 0.080000000000000002,
         0.040000000000000001, -0.02])
    np.testing.assert_array_equal(
        np.asarray(THISq.excess_dfe_cursors),
        [0, 0, 0.009999999999999995, 0, 0])


def test_oracle_c2m_window():
    """do_C2M shifts the window by T_O; excess is windowed minus actual."""
    THIS, _ = OptFom_Compute_DFE(_oct_sbr(),
                                 SimpleNamespace(cursor_i=_OCT_CUR),
                                 _oct_param(), True, 2)
    np.testing.assert_array_equal(
        np.asarray(THIS.excess_dfe_cursors),
        [-0.29999999999999999, 0.17999999999999999, -0.089999999999999997,
         -0.040000000000000001, 0.02])


def test_oracle_floating_dfe_branch():
    """Floating_DFE=1 with ndfe=20, ndfe_passed=5, N_bmax=20."""
    THIS, param = OptFom_Compute_DFE(
        _oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
        _oct_param(ndfe=20, bmax=np.full(5, 0.5), bmin=-np.full(5, 0.5),
                   **_OCT_FLOAT), False, 0)
    np.testing.assert_array_equal(
        np.asarray(THIS.floating_tap_locations).ravel(),
        [6, 7, 8, 9, 10, 11, 12, 13])
    np.testing.assert_allclose(
        np.asarray(THIS.dfetaps)[:13],
        [0.29999999999999999, -0.17999999999999999, 0.089999999999999997,
         0.040000000000000001, -0.02, 0.013107200000000006,
         -0.010485760000000004, 0.008388608000000004, -0.0067108864000000037,
         0.0053687091200000031, -0.0042949672960000025, 0.0034359738368000023,
         -0.0027487790694400022], rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(param.use_bmax).ravel(),
        [0.5] * 5 + [0.050000000000000003] * 8 + [0.0] * 7, rtol=0, atol=0)


def test_oracle_floating_dfe_with_tail_rss():
    """Floating taps and the RSS rescale together."""
    THIS, _ = OptFom_Compute_DFE(
        _oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
        _oct_param(ndfe=20, bmax=np.full(5, 0.5), bmin=-np.full(5, 0.5),
                   N_tail_start=3, B_float_RSS_MAX=0.05, **_OCT_FLOAT),
        False, 0)
    # 1 ulp: Octave's norm() of these 20 taps gives ...303 where every numpy
    # spelling (linalg.norm, sqrt(sum(v*v)), sqrt(v@v)) gives ...302.  That is
    # a BLAS difference, not a form difference; the 5-tap cases above are
    # bit-exact.
    assert THIS.tail_RSS == pytest.approx(0.10278028059573303, rel=1e-15)
    # rtol=1e-15 rather than 0 for the same reason: that one ulp of tail_RSS
    # divides straight into the rescaled taps.
    np.testing.assert_allclose(
        np.asarray(THIS.dfetaps)[:5],
        [0.29999999999999999, -0.17999999999999999, 0.043782717598329059,
         0.019458985599257364, -0.009729492799628682], rtol=1e-15)


def test_oracle_dfecursors_past_the_end_is_an_error():
    """COM Octave, ndfe=40 on a 300-sample sbr:
    "error: sbr(361): out of bound 300 (dimensions are 1x300)".
    The Python slice truncated to 32 cursors instead."""
    with pytest.raises(IndexError, match=r'sbr\(361\).*out of bound 300'):
        OptFom_Compute_DFE(_oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
                           _oct_param(ndfe=40, bmax=np.full(40, 0.5),
                                      bmin=-np.full(40, 0.5)), False, 0)


def test_oracle_c2m_window_before_the_start_is_an_error():
    """COM Octave, cursor_i=5 (1-based) with T_O=20:
    "error: sbr(-7): subscripts must be either integers 1 to (2^63)-1 or
    logicals".  A Python slice with a negative start read from the tail."""
    with pytest.raises(IndexError, match=r'sbr\(-7\)'):
        OptFom_Compute_DFE(_oct_sbr(cursor=4), SimpleNamespace(cursor_i=4),
                           _oct_param(), True, 20)


def test_oracle_negative_n_tail_start_is_an_error():
    """COM Octave, param.N_tail_start=-2:
    "error: dfetaps(-2): subscripts must be either integers 1 to (2^63)-1 or
    logicals".  dfetaps[N_tail_start-1:] silently took the last three taps and
    reported tail_RSS=0.10049875621120891 for them."""
    with pytest.raises(IndexError, match=r'dfetaps\(-2\)'):
        OptFom_Compute_DFE(_oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
                           _oct_param(N_tail_start=-2), False, 0)


def test_oracle_fractional_t_o_is_refused():
    """T_O=2.5 makes the window start 46.5, which MATLAB refuses.

    Octave takes a non-integer range as an index with
    "warning: non-integer range used as index" and truncates; that extension
    is not pinned.  What matters is that the port no longer answers.
    """
    with pytest.raises(ValueError, match=r'sbr\(46.5\)'):
        OptFom_Compute_DFE(_oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
                           _oct_param(), True, 2.5)


def test_oracle_fractional_ndfe_truncates_the_same_way():
    """ndfe=5.5 still selects five cursors, because they step by M.

    COM Octave agrees with ndfe=5 here; pinned so the switch away from
    int(param.ndfe) is shown not to have moved this case.
    """
    THIS, _ = OptFom_Compute_DFE(
        _oct_sbr(), SimpleNamespace(cursor_i=_OCT_CUR),
        _oct_param(bmax=np.full(5, 0.5), bmin=-np.full(5, 0.5), ndfe=5.5),
        False, 0)
    np.testing.assert_array_equal(
        np.asarray(THIS.dfetaps),
        [0.29999999999999999, -0.17999999999999999, 0.089999999999999997,
         0.040000000000000001, -0.02])


def test_oracle_zero_cursor_gives_nan_taps():
    """sbr(cursor_i)=0 divides through by zero: COM Octave returns NaN taps
    and zero excess rather than raising."""
    THIS, _ = OptFom_Compute_DFE(np.zeros(300),
                                 SimpleNamespace(cursor_i=_OCT_CUR),
                                 _oct_param(), False, 0)
    assert np.all(np.isnan(np.asarray(THIS.dfetaps)))
    np.testing.assert_array_equal(np.asarray(THIS.excess_dfe_cursors),
                                  np.zeros(5))
