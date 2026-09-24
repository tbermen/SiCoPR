# ============================================================
# MATLAB GROUND TRUTH
# COM_eye_width: computes eye contour and width for COM.
# MATLAB lines 1362–1551
#
# Key invariants:
# 1. eye_contour shape = (samp_UI, 2*(levels-1))
#    eye_contour[:,0] = A_ni_top of eye 0 (level 1's top)
#    eye_contour[:,1] = A_ni_bot of eye 0 (level 0's bottom)
#
# 2. Left_EW and Right_EW have length (levels-1)
#
# 3. Levels vector: [-1, -1/3, 1/3, 1] for L=4
#    A_s_vec shift: pdf_full{n}[j].x += A_s_vec[j] * Levels[n]
#    For n=0 (level -1), shift is negative; n=3 (level +1), shift is positive
#
# 4. out_VT, out_VB are empty when T_O=0
#
# 5. When T_O != 0 (windowed), out_VT and out_VB are set to the worst eye's
#    top and bottom voltage respectively
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace

from com_functions.fn.COM_eye_width.py_impl import COM_eye_width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(samp_UI=16, levels=4, T_O=0):
    return SimpleNamespace(
        samples_for_C2M=samp_UI, T_O=T_O, levels=levels, specBER=1e-4,
        sigma_RJ=0.01, sigma_X=1.0, A_DD=0.05, QL=2.5,
    )


def _make_noise():
    delta_y = 0.01
    ne_pdf = SimpleNamespace(x=np.array([0.0]), y=np.array([1.0]), BinSize=delta_y, Min=0)
    cci_pdf = SimpleNamespace(x=np.array([0.0]), y=np.array([1.0]), BinSize=delta_y, Min=0)
    return SimpleNamespace(
        sigma_N=0.01, sigma_TX=0.005, ber_q=7.0,
        ne_noise_pdf=ne_pdf, cci_pdf=cci_pdf,
    )


def _make_op():
    return SimpleNamespace(Histogram_Window_Weight='rectangle', ber_q=7.0)


def _make_fom(samp_UI=16):
    return SimpleNamespace(t_s=samp_UI // 2)


def _stub_get_pdf_full(chdata_0, delta_y, t_s, param, OP, pdf_range):
    samp_UI = int(param.samples_for_C2M)
    n_bins = 16
    x0 = np.linspace(-0.5, 0.5, n_bins)
    dx = x0[1] - x0[0]
    y0 = np.ones(n_bins) / n_bins
    pdf_list = [SimpleNamespace(x=x0.copy(), y=y0.copy(), BinSize=dx, Min=-n_bins//2)
                for _ in range(samp_UI)]
    h_j_full = np.zeros((2, samp_UI))
    A_s_vec = np.ones(samp_UI) * 0.3
    return pdf_list, h_j_full, A_s_vec


# ---------------------------------------------------------------------------
# Test 1 – Nominal: eye_contour shape is correct
# ---------------------------------------------------------------------------

def test_eye_contour_shape():
    """eye_contour.shape = (samp_UI, 2*(levels-1))."""
    samp_UI = 16
    levels = 4
    param = _make_param(samp_UI=samp_UI, levels=levels)
    ch = SimpleNamespace()
    delta_y = 0.01

    Left_EW, Right_EW, eye_contour, out_VT, out_VB = COM_eye_width(
        [ch], delta_y, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    assert eye_contour.shape == (samp_UI, 2 * (levels - 1)), \
        f"eye_contour.shape={eye_contour.shape}, expected ({samp_UI}, {2*(levels-1)})"


# ---------------------------------------------------------------------------
# Test 2 – Nominal: Left_EW and Right_EW have correct length
# ---------------------------------------------------------------------------

def test_ew_lengths():
    """Left_EW and Right_EW each have length (levels-1)."""
    samp_UI = 16
    levels = 4
    param = _make_param(samp_UI=samp_UI, levels=levels)
    ch = SimpleNamespace()

    Left_EW, Right_EW, _, _, _ = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    assert len(Left_EW) == levels - 1
    assert len(Right_EW) == levels - 1


# ---------------------------------------------------------------------------
# Test 3 – Nominal: out_VT, out_VB are empty when T_O = 0
# ---------------------------------------------------------------------------

def test_out_vt_vb_empty_when_T_O_zero():
    """When param.T_O=0, out_VT and out_VB should be [] (not set)."""
    samp_UI = 16
    param = _make_param(samp_UI=samp_UI, T_O=0)
    ch = SimpleNamespace()

    _, _, _, out_VT, out_VB = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    assert out_VT == [] or out_VT is None or out_VT == 0.0 or not out_VT, \
        f"out_VT should be empty, got {out_VT}"


# ---------------------------------------------------------------------------
# Test 4 – Boundary: pdf_range_flag=True limits computation to T_O window
# ---------------------------------------------------------------------------

def test_pdf_range_flag_limits_computation():
    """pdf_range_flag=True → PDF only built for center ± T_O samples."""
    samp_UI = 16
    T_O_val = 2  # percent → T_O_samples = floor(2/1000 * 16) = 0 ... use larger value
    # Use T_O=125 (‰) to get T_O_samples = floor(0.125 * 16) = 2
    param = _make_param(samp_UI=samp_UI, T_O=125)
    ch = SimpleNamespace()

    # Track which j values are passed to pdf_full stub via get_pdf_full
    passed_ranges = {}

    def track_pdf_full(chdata_0, delta_y, t_s, p, op, pdf_range):
        passed_ranges['pdf_range'] = pdf_range
        return _stub_get_pdf_full(chdata_0, delta_y, t_s, p, op, pdf_range)

    Left_EW, Right_EW, ec, _, _ = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), True,
        _get_pdf_full_fn=track_pdf_full)

    assert 'pdf_range' in passed_ranges
    pr = passed_ranges['pdf_range']
    assert len(pr) == 2, f"Expected [start, end] range, got {pr}"


# ---------------------------------------------------------------------------
# Test 5 – Nominal: 2-level PAM (levels=2) → 1 eye
# ---------------------------------------------------------------------------

def test_pam2_single_eye():
    """For levels=2, there is 1 eye → Left_EW/Right_EW each length 1."""
    samp_UI = 16
    param = _make_param(samp_UI=samp_UI, levels=2)
    ch = SimpleNamespace()

    def stub_pdf2(chdata_0, delta_y, t_s, p, op, pr):
        n = samp_UI
        nb = 16
        x0 = np.linspace(-0.5, 0.5, nb)
        dx = x0[1] - x0[0]
        y0 = np.ones(nb) / nb
        pdf_list = [SimpleNamespace(x=x0.copy(), y=y0.copy(), BinSize=dx, Min=-nb//2)
                    for _ in range(n)]
        h_j = np.zeros((1, n))
        A_s = np.ones(n) * 0.3
        return pdf_list, h_j, A_s

    Left_EW, Right_EW, ec, _, _ = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=stub_pdf2)

    assert len(Left_EW) == 1
    assert ec.shape == (samp_UI, 2)


# ---------------------------------------------------------------------------
# Test 6 – Windowed: out_VT and out_VB set when T_O != 0
# ---------------------------------------------------------------------------

def test_out_vt_vb_set_when_T_O_nonzero():
    """When T_O != 0, out_VT and out_VB are scalars (not empty)."""
    samp_UI = 16
    param = _make_param(samp_UI=samp_UI, T_O=125)  # T_O_samples = 2
    ch = SimpleNamespace()

    _, _, _, out_VT, out_VB = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    # out_VT and out_VB should now be numeric scalars
    assert isinstance(out_VT, (int, float, np.floating)), \
        f"out_VT should be scalar, got {type(out_VT)}"
    assert isinstance(out_VB, (int, float, np.floating)), \
        f"out_VB should be scalar, got {type(out_VB)}"


# ---------------------------------------------------------------------------
# Bathtub BER composition (added 2026-08-19).
#
# pdf_to_cdf returns THREE curves and picking the wrong one fails silently:
#     yB = cumsum(pdf)     = P(V <= v)   bottom-eye tail
#     yT = reverse cumsum  = P(V >= v)   top-eye tail
#     y  = min(yB, yT)     -- necessarily ~0.5 mid-distribution
#
# The timing/voltage bathtubs must use yB for the UPPER level (it errs by
# falling below the threshold) and yT for the LOWER level (it errs by rising
# above it). An earlier version used `y` and `1 - y`, which pinned every
# bathtub at BER ~ 0.5 no matter how open the eye was -- a link with COM
# 5.6 dB plotted as if it were closed.
#
# Two Gaussian levels with a threshold midway between them have a known answer,
# so this checks the composition against theory rather than against itself.
# ---------------------------------------------------------------------------
def _gauss_pdf(centre, sigma, x, binsize):
    y = np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    return SimpleNamespace(x=x, y=y / y.sum(), BinSize=binsize,
                           Min=int(x[0] / binsize))


def test_bathtub_ber_matches_gaussian_theory():
    """0.5*(yB_up + yT_lo) at the midpoint must equal the Q-function BER."""
    import math
    from com_functions.fn.pdf_to_cdf.py_impl import pdf_to_cdf

    mu, sigma, bs = 0.030, 0.004, 1e-5
    x = np.arange(-0.12, 0.12 + bs, bs)
    cdf_up = pdf_to_cdf(_gauss_pdf(+mu, sigma, x, bs))
    cdf_lo = pdf_to_cdf(_gauss_pdf(-mu, sigma, x, bs))

    ber = 0.5 * (np.interp(0.0, cdf_up.x, cdf_up.yB)
                 + np.interp(0.0, cdf_lo.x, cdf_lo.yT))
    analytic = 0.5 * math.erfc(mu / (sigma * math.sqrt(2)))
    assert 0.9 < ber / analytic < 1.1, (
        'bathtub BER %.4e vs analytic %.4e' % (ber, analytic))


def test_bathtub_must_not_use_the_min_curve():
    """The discarded formulation must be demonstrably wrong, or the test above
    could pass for the wrong reason."""
    from com_functions.fn.pdf_to_cdf.py_impl import pdf_to_cdf

    mu, sigma, bs = 0.030, 0.004, 1e-5
    x = np.arange(-0.12, 0.12 + bs, bs)
    cdf_up = pdf_to_cdf(_gauss_pdf(+mu, sigma, x, bs))
    cdf_lo = pdf_to_cdf(_gauss_pdf(-mu, sigma, x, bs))

    wrong = 0.5 * (np.interp(0.0, cdf_up.x, cdf_up.y)
                   + (1.0 - np.interp(0.0, cdf_lo.x, cdf_lo.y)))
    assert wrong > 0.4, (
        'the y/1-y formulation no longer collapses to ~0.5 (got %.3e); if '
        'pdf_to_cdf changed, revisit the bathtub composition' % wrong)


def test_pdf_to_cdf_tail_semantics():
    """yB rises with v, yT falls with v, and y is their minimum."""
    from com_functions.fn.pdf_to_cdf.py_impl import pdf_to_cdf

    bs = 1e-4
    x = np.arange(-0.05, 0.05 + bs, bs)
    c = pdf_to_cdf(_gauss_pdf(0.0, 0.01, x, bs))
    assert np.all(np.diff(c.yB) >= -1e-15), 'yB must be non-decreasing'
    assert np.all(np.diff(c.yT) <= 1e-15), 'yT must be non-increasing'
    np.testing.assert_allclose(c.y, np.minimum(c.yB, c.yT))
    assert abs(c.y.max() - 0.5) < 0.02, (
        'min(yB, yT) should peak near 0.5 -- this is exactly why it cannot be '
        'used as a BER')


# --------------------------------------------------------------------------
# OP.Histogram_Window_Weight, against COM Octave.
#
# 'gaussian', 'triangle' and 'dual_rayleigh' sat in test_option_coverage.py's
# KNOWN_UNCOVERED. They were unreachable in practice: the window was built
# inline in COM_eye_width, which needs the whole get_pdf_full chain to run, so
# no test could select a window type and observe it. The block is now the
# module-level helper _histogram_window, which is the entire behaviour of the
# option and can be driven directly.
#
# Expected values are COM Octave's, produced by lifting ML 1502-1524 VERBATIM
# out of matlab/com_ieee8023_4p16p0.m into an Octave function and running it on
# the same (T_O, QL) -- lifted rather than retyped, so a transcription slip
# cannot masquerade as agreement. All four types agree at T_O = 3, 5, 7, 8 and
# QL = 1.5, 2.5, 4.0.
# --------------------------------------------------------------------------

from com_functions.fn.COM_eye_width.py_impl import _histogram_window  # noqa: E402

_OCT_WINDOW = {
    'gaussian': [
        0.04393693362340742, 0.1353352832366127, 0.32465246735834974,
        0.60653065971263342, 0.88249690258459546, 1.0, 0.88249690258459546,
        0.60653065971263342, 0.32465246735834974, 0.1353352832366127,
        0.04393693362340742],
    'triangle': [
        0.0, 0.20000000000000001, 0.40000000000000002, 0.60000000000000009,
        0.80000000000000004, 1.0, 0.80000000000000004, 0.60000000000000009,
        0.40000000000000002, 0.19999999999999996, 0.0],
    'dual_rayleigh': [
        3.065324644063862e-05, 0.72618639222182202, 1.0, 0.81371490177881944,
        0.50010084511932928, 0.36139924806929608, 0.50010084511932928,
        0.81371490177881944, 1.0, 0.72618639222182202, 3.065324644063862e-05],
    'rectangle': [1.0] * 11,
}


@pytest.mark.parametrize('hw_type', sorted(_OCT_WINDOW))
def test_histogram_window_matches_com_octave(hw_type):
    """T_O=5, QL=2.5. The triangle window comes from a MATLAB colon expression,
    which accumulates differently from arange, so it is held to 1e-15 rather
    than exactly; the other three are bit-exact."""
    got = np.asarray(_histogram_window(5, 2.5, hw_type), dtype=float)
    want = np.asarray(_OCT_WINDOW[hw_type], dtype=float)
    assert got.size == want.size, (
        '%s gives %d weights, COM Octave gives %d'
        % (hw_type, got.size, want.size))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


def test_unrecognised_window_type_is_refused():
    """ML 1523: otherwise -> error('%s not recognized for
    Histogram_Window_Weight'). The port used to fall through to the rectangle
    window, silently answering on a misspelled option where the reference
    stops."""
    with pytest.raises(ValueError, match='not recognized'):
        _histogram_window(5, 2.5, 'gausian')


def test_window_weights_are_symmetric_and_peak_at_one():
    """Guard the guard: a window that had collapsed to all-ones would still
    match 'rectangle' above, so check the three shaped windows really are
    shaped."""
    for hw_type in ('gaussian', 'triangle', 'dual_rayleigh'):
        w = np.asarray(_histogram_window(5, 2.5, hw_type), dtype=float)
        np.testing.assert_allclose(w, w[::-1], rtol=0, atol=1e-15)
        assert abs(float(np.max(w)) - 1.0) < 1e-12, hw_type
        assert float(np.min(w)) < 0.99, '%s is flat, not shaped' % hw_type


def test_stub_convolution_is_direct():
    """The stub _conv1d convolves directly, as conv2 does; its FFT dispatch at
    128 bins was the shortcut removed from conv_fct in 8ec85b0."""
    import com_functions.fn.COM_eye_width.py_impl as m
    assert not hasattr(m, '_CONV_FFT_MIN')
    a = np.exp(-np.arange(300.0) / 3)
    np.testing.assert_array_equal(m._conv1d(a, a), np.convolve(a, a))
