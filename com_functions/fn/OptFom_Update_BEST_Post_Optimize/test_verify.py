"""Verification tests for OptFom_Update_BEST_Post_Optimize().

# ============================================================
# MATLAB GROUND TRUTH (lines 3843-3890)
# Updates BEST after optimization: cursor value, CTLE gains,
# precursor/postcursor samples, DFE tap clipping.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Update_BEST_Post_Optimize.py_impl import OptFom_Update_BEST_Post_Optimize


def _f(N=50):
    return np.linspace(0, 50e9, N)


def _sbr(M=8, peak=40, N=200):
    sbr = np.zeros(N)
    sbr[peak] = 1.0
    for k in range(1, 4):
        sbr[peak + k * M] = 0.3 / k
    return sbr


def _param(M=8):
    return SimpleNamespace(
        samples_per_ui=M,
        ui=1 / 25e9,
        ndfe=3,
        ndfe_passed=3,
        CTLE_type='CL93',
        CTLE_fz=np.array([1e9]),
        CTLE_fp1=np.array([5e9]),
        CTLE_fp2=np.array([10e9]),
        Floating_DFE=False,
        fb=25e9,
        fb_BW_cutoff=1.0,
        BTorder=2,
        fb_BT_cutoff=1.0,
        RC_Start=5e9,
        RC_end=25e9,
    )


def _op():
    return SimpleNamespace(
        Bessel_Thomson=False,
        Butterworth=False,
        Raised_Cosine=False,
    )


def _best(M=8, cursor_i=40, N=200):
    return SimpleNamespace(
        sbr=_sbr(M=M, peak=cursor_i, N=N),
        cursor_i=cursor_i,
        ctle=1,            # 1-BASED; 0 is out of bounds in MATLAB
        gdc=-6.0,
        G_high_pass=1,     # 1-BASED
        bmax=np.array([0.9, 0.9, 0.9]),
        bmin=np.array([-0.9, -0.9, -0.9]),
    )


def test_cursor_value_set():
    """BEST.cursor = sbr at cursor_i."""
    B = _best(cursor_i=40)
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(), _op())
    assert result.cursor == pytest.approx(float(B.sbr[40]))


def test_h_r_is_array():
    """BEST.H_r is a numpy array of same length as f."""
    B = _best()
    f = _f(50)
    result = OptFom_Update_BEST_Post_Optimize(B, f, _param(), _op())
    assert len(result.H_r) == len(f)


def test_ctle_gain_is_complex_array():
    """BEST.ctle_gain is complex-valued."""
    B = _best()
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(), _op())
    assert np.iscomplexobj(result.ctle_gain)


def test_precursors_before_cursor():
    """sampled_sbr_precursors are the UI-SPACED sbr samples before the cursor.

    MATLAB's (cursor_i/M : -1 : 1/M) steps by a whole UI, so with a 1-based
    cursor at 41 and M=8 the precursors are 1-based indices 1, 9, 17, 25, 33.
    """
    B = _best(cursor_i=40)          # 0-based; MATLAB's cursor_i is 41
    sbr = B.sbr.copy()
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(), _op())
    np.testing.assert_array_equal(result.sampled_sbr_precursors,
                                  sbr[[0, 8, 16, 24, 32]])


def test_postcursors_correct_length():
    """sampled_sbr_postcursors has the right number of elements (UI-spaced after cursor)."""
    M = 8
    cursor_i = 40
    N = 200
    B = _best(M=M, cursor_i=cursor_i, N=N)
    result = OptFom_Update_BEST_Post_Optimize(B, _f(), _param(M), _op())
    expected_count = len(range(cursor_i + M, N, M))
    assert len(result.sampled_sbr_postcursors) == expected_count


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py;
# OptFom_Update_BEST_Post_Optimize, OptFom_Calc_Hr, the three filters,
# Tukey_Window, bessel, FD_CTLE and dfe_clipper are all byte-identical
# between octave/com_ieee8023_4p16p0_octave_compat.m and
# matlab/com_ieee8023_4p16p0.m).
#
# Fixture: 200-sample sbr, samples_per_ui=8, ui=1/53.125e9, cursor_i=33
# (1-based; BEST.cursor_i below is the 0-based 32), ndfe=4, CTLE_type
# 'CL120d', ctle=2, G_high_pass=2, gdc=-6, bmax=[.7 .35 .2 .15 .1 .08],
# bmin=-bmax, Butterworth/Bessel-Thomson/Raised-Cosine all on.
#
# What these catch:
#   1. `(cursor_i/M : -1 : 1/M)` steps by a whole UI, so the precursors are
#      UI-spaced. The port returned every sample from 0 to cursor_i-1 -- 32
#      of them where MATLAB gives 4 -- with times to match.
#   2. sampled_sbr_postcursors_t is the 1-BASED index over M; the port used
#      the 0-based one and every time came out ui/M short.
#   3. `BEST.sampled_sbr_postcursors(1:param.ndfe)` is a read, so too few
#      postcursors is an error, not a shorter tap set.
#   4. The private copy of OptFom_Calc_Hr that used to live here was the
#      pre-oracle form; its np.where Tukey window is wrong for any f that
#      does not ascend.
# ============================================================

_OM = 8
_OUI = 1.0 / 53.125e9
_ON = np.arange(200)
_OSBR = (np.exp(-((_ON - 32.0) / 6.0) ** 2) * 0.9
         - 0.08 * np.exp(-((_ON - 70.0) / 14.0) ** 2)
         + 0.02 * np.sin(_ON / 5.0))
_OBMAX = np.array([0.7, 0.35, 0.2, 0.15, 0.1, 0.08])


def _oracle_best(cursor_i0=32):
    return SimpleNamespace(
        sbr=_OSBR.copy(), cursor_i=cursor_i0, ctle=2, gdc=-6.0,
        G_high_pass=2, bmax=_OBMAX.copy(), bmin=-_OBMAX.copy(),
        floating_tap_locations=np.array([2, 4]))


def _oracle_param(ndfe=4):
    return SimpleNamespace(
        samples_per_ui=_OM, ui=_OUI, ndfe=ndfe, ndfe_passed=ndfe,
        CTLE_type='CL120d', Floating_DFE=False,
        CTLE_fz=np.array([5e9, 5.5e9, 6e9, 6.5e9]),
        CTLE_fp1=np.array([10e9, 11e9, 12e9, 13e9]),
        CTLE_fp2=np.array([20e9, 21e9, 22e9, 23e9]),
        f_HP=np.array([0.6e9, 0.7e9, 0.8e9, 0.9e9]),
        g_DC_HP_values=np.array([0.0, -0.5, -1.0, -1.5]),
        f_HP_Z=np.array([0.5e9, 0.6e9, 0.7e9, 0.8e9]),
        f_HP_P=np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9]),
        fb=53.125e9, fb_BW_cutoff=0.75, BTorder=4, fb_BT_cutoff=0.75,
        RC_Start=20e9, RC_end=40e9)


def _oracle_op():
    return SimpleNamespace(Butterworth=1, Bessel_Thomson=1, Raised_Cosine=1)


def test_octave_precursors_are_ui_spaced():
    """COM Octave, cursor_i=33, M=8: four precursors, not 32."""
    B = OptFom_Update_BEST_Post_Optimize(
        _oracle_best(), np.linspace(0.0, 40e9, 17), _oracle_param(),
        _oracle_op())
    np.testing.assert_allclose(B.sampled_sbr_precursors_t, [
        2.3529411764705883e-12, 2.1176470588235294e-11,
        4.0000000000000004e-11, 5.8823529411764704e-11], rtol=1e-14)
    np.testing.assert_allclose(B.sampled_sbr_precursors, [
        -7.1203151202759749e-13, 0.01999157309949072,
        -0.00043312147695474855, 0.13218705305379491], rtol=1e-12)


def test_octave_postcursor_times_use_the_one_based_index():
    """COM Octave: the first postcursor sits at 41/8*ui, not 40/8*ui."""
    B = OptFom_Update_BEST_Post_Optimize(
        _oracle_best(), np.linspace(0.0, 40e9, 17), _oracle_param(),
        _oracle_op())
    assert len(B.sampled_sbr_postcursors_t) == 20
    np.testing.assert_allclose(B.sampled_sbr_postcursors_t[:6], [
        9.6470588235294118e-11, 1.1529411764705883e-10,
        1.3411764705882354e-10, 1.5294117647058824e-10,
        1.7176470588235294e-10, 1.9058823529411764e-10], rtol=1e-14)
    # the DFE cursor times coincide with the first ndfe postcursor times
    np.testing.assert_allclose(B.sampled_sbr_dfecursors_t,
                               B.sampled_sbr_postcursors_t[:4], rtol=1e-14)
    np.testing.assert_allclose(B.DFE_taps_mV, [
        0.17108841060740826, -0.0095230794933926152,
        -0.049013808595084488, -0.061946403552794321], rtol=1e-12)
    assert B.cursor == pytest.approx(0.90228046077887869, rel=1e-14)


def test_octave_too_few_postcursors_is_an_error():
    """COM Octave, ndfe=30 with 20 postcursors:
      "error: BEST(30): out of bound 20 (dimensions are 1x20)".
    min(ndfe, n_post) answered with a shorter tap set instead.
    """
    with pytest.raises(IndexError, match='out of bound'):
        OptFom_Update_BEST_Post_Optimize(
            _oracle_best(), np.linspace(0.0, 40e9, 17), _oracle_param(ndfe=30),
            _oracle_op())


def test_octave_ctle_index_zero_is_out_of_bounds():
    """COM Octave, BEST.ctle=0: "param(0): subscripts must be either integers
    1 to (2^63)-1 or logicals". Python's [-1] read the LAST CTLE setting.
    """
    B = _oracle_best()
    B.ctle = 0
    with pytest.raises(IndexError, match='subscripts'):
        OptFom_Update_BEST_Post_Optimize(
            B, np.linspace(0.0, 40e9, 17), _oracle_param(), _oracle_op())


def test_octave_cl120e_h_low_is_indexed_by_ctle():
    """COM Octave, the fixture above with param.CTLE_type = 'CL120e'.

    CL120e is `FD_CTLE(f, param.f_HP_Z(BEST.ctle), param.f_HP_P(BEST.ctle),
    100e100, 0)`: the high-pass zero and pole come from the CTLE index, not
    from G_high_pass as CL120d's do, and the DC term is a literal 0 dB, not
    BEST.gdc.  BEST.ctle is 2 and BEST.G_high_pass is 4 here, and f_HP_Z and
    f_HP_P hold distinct values, so a lookup by the wrong index, a swapped
    zero and pole, or gdc in place of 0 all give a different H_low.
    (100e100 is 1e102; the port writes 1e200. At these frequencies both make
    the second pole term exactly 1.0 in double, so the results are bit-equal
    and Octave confirms it -- every field of BEST agreed.)
    """
    B = _oracle_best()
    B.G_high_pass = 4
    p = _oracle_param()
    p.CTLE_type = 'CL120e'
    B = OptFom_Update_BEST_Post_Optimize(B, np.linspace(0.0, 40e9, 17), p,
                                         _oracle_op())
    np.testing.assert_allclose(B.H_low, [
        1 + 0j,
        1.6981680071492407 + 0.30719392314566574j,
        1.7948620119547249 + 0.1748696426300394j,
        1.8157848938391925 + 0.11964845109641489j,
        1.823370549682179 + 0.090570760465039674j,
        1.8269295905838521 + 0.072769803971378999j,
        1.8288758233499844 + 0.060784227045665523j,
        1.8300537739326528 + 0.052174808647195314j,
        1.8308201025232007 + 0.045695105638776032j,
        1.8313463130098926 + 0.040643597524928084j,
        1.831723117378089 + 0.03659581716463594j,
        1.8320021299254527 + 0.033280085197018111j,
        1.8322144672162983 + 0.030514530464597595j,
        1.8323797905673342 + 0.028172854449971314j,
        1.8325110163294489 + 0.02616463194178268j,
        1.8326169127364187 + 0.02442342944026828j,
        1.8327036012348994 + 0.022899349033959727j], rtol=1e-12)
    # and it reaches ctle_gain = H_low .* ctle_gain1 .* H_r
    np.testing.assert_allclose(
        [B.ctle_gain[1], B.ctle_gain[8]],
        [1.0630012605068855 + 0.38421718687499395j,
         -1.4241087791559466 - 1.8052913027879163j], rtol=1e-12)


def test_octave_h_r_on_a_non_ascending_faxis():
    """Tukey_Window concatenates three counted pieces, so H_r is grouped by
    category and only lines up with f when f ascends.

    COM Octave, f = [25 0 35 15 40 20 45] GHz with RC_Start=20e9,
    RC_end=40e9: H_r(1) = -0.69940492382039898-0.65853321494420236j. The
    private np.where copy gave -0.59697944412453186-0.562093258433913j.
    """
    f = np.array([25e9, 0.0, 35e9, 15e9, 40e9, 20e9, 45e9])
    B = OptFom_Update_BEST_Post_Optimize(_oracle_best(), f, _oracle_param(),
                                         _oracle_op())
    np.testing.assert_allclose(B.H_r, [
        -0.69940492382039898 - 0.65853321494420236j,
        1 + 0j,
        -0.63579024836023201 + 0.27726128317245896j,
        0.027276115775619132 - 0.1423496081611545j,
        0j,
        -0.28751805518656903 - 0.93700563475707754j,
        0j], rtol=1e-12, atol=1e-16)
