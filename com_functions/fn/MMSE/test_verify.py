"""Tests for MMSE (MATLAB lines ~2480).

MATLAB GROUND TRUTH:
  Basic MMSE with no floating taps (N_bg=0):
    sbr pulse of length num_ui, cursor at center → FOM finite, sigma_e > 0
"""
import numpy as np
import pytest
from com_functions.fn.MMSE.py_impl import MMSE
from types import SimpleNamespace


def _param():
    p = SimpleNamespace()
    p.num_ui_RXFF_noise = 20
    p.samples_per_ui = 4
    p.levels = 4
    p.fb = 26.5625e9
    p.ndfe = 2
    p.N_bg = 0
    p.N_bf = 1
    p.N_bmax = 4
    p.RxFFE_cmx = 2
    p.RxFFE_cpx = 2
    p.bmax = np.array([0.9, 0.9])
    p.bmin = np.array([-0.9, -0.9])
    p.ffe_tapn_max = 1.0
    p.ffe_pre_tap1_max = 1.0
    p.ffe_post_tap1_max = 1.0
    p.R_LM = 1.0
    p.bmaxg = 1.0
    return p


def _psd(n_f):
    PSD = SimpleNamespace()
    PSD.S_n = np.ones(n_f) * 1e-4
    PSD.S_isi = np.zeros(n_f)
    return PSD


def test_basic_no_floating():
    p = _param()
    M = p.samples_per_ui
    num_ui = p.num_ui_RXFF_noise
    N = num_ui * M
    sbr = np.zeros(N)
    cursor_i = (num_ui // 2) * M
    sbr[cursor_i] = 1.0
    n_f = N // 2 + 1
    PSD = _psd(n_f)
    r = MMSE(PSD, sbr, cursor_i, p, SimpleNamespace(RXFFE_FLOAT_CTL='isi'))
    assert r.sigma_e >= 0
    assert np.isfinite(r.FOM)
    assert len(r.C) > 0


def test_returns_required_fields():
    p = _param()
    M = p.samples_per_ui
    num_ui = p.num_ui_RXFF_noise
    N = num_ui * M
    sbr = np.zeros(N)
    cursor_i = (num_ui // 2) * M
    sbr[cursor_i] = 1.0
    n_f = N // 2 + 1
    PSD = _psd(n_f)
    OP = SimpleNamespace(RXFFE_FLOAT_CTL='isi')
    r = MMSE(PSD, sbr, cursor_i, p, OP)
    for field in ('sigma_e', 'FOM', 'C', 'floating_tap_locations', 'blim', 'Nw'):
        assert hasattr(r, field), f'missing field: {field}'


def test_cursor_alignment():
    p = _param()
    M = p.samples_per_ui
    num_ui = p.num_ui_RXFF_noise
    N = num_ui * M
    sbr = np.zeros(N)
    cursor_i = M * 3  # not centered
    sbr[cursor_i] = 1.0
    n_f = N // 2 + 1
    PSD = _psd(n_f)
    OP = SimpleNamespace(RXFFE_FLOAT_CTL='isi')
    r = MMSE(PSD, sbr, cursor_i, p, OP)
    assert np.isfinite(r.FOM)


@pytest.mark.parametrize('taps', [
    {0: 1.0},                                    # a bare impulse
    {-1: 0.35, 0: 1.0, 1: 0.6, 2: 0.3},          # ISI, offsets in UI
], ids=['impulse', 'isi'])
def test_c_normalized(taps):
    p = _param()
    M = p.samples_per_ui
    num_ui = p.num_ui_RXFF_noise
    N = num_ui * M
    sbr = np.zeros(N)
    cursor_i = (num_ui // 2) * M
    for k, v in taps.items():
        sbr[cursor_i + k * M] = v
    n_f = N // 2 + 1
    PSD = _psd(n_f)
    OP = SimpleNamespace(RXFFE_FLOAT_CTL='isi')
    r = MMSE(PSD, sbr, cursor_i, p, OP)
    dw = p.RxFFE_cmx
    # ML 84: Craw = w/w(dw+1), and with no floating taps C = Craw (ML 91). The
    # main tap is w(dw+1)/w(dw+1), which is exactly 1.0 in IEEE arithmetic for
    # any finite non-zero w(dw+1) -- not "about 1". This line used to read
    # `< 0.1 or True`, which could never fail.
    #
    # The impulse alone cannot tell normalising from not normalising: its raw
    # main tap is already 1. The ISI channel's is not, so between them the
    # test catches a dropped normalisation and the 1-based index slip
    # (w[dw+1] for w(dw+1)).
    assert r.C[dw] == 1.0, 'main tap is %r; ML 84 makes it exactly 1' % r.C[dw]


def test_with_isi():
    p = _param()
    M = p.samples_per_ui
    num_ui = p.num_ui_RXFF_noise
    N = num_ui * M
    rng = np.random.default_rng(7)
    sbr = rng.standard_normal(N) * 0.1
    cursor_i = (num_ui // 2) * M
    sbr[cursor_i] = 1.0
    n_f = N // 2 + 1
    PSD = _psd(n_f)
    OP = SimpleNamespace(RXFFE_FLOAT_CTL='isi')
    r = MMSE(PSD, sbr, cursor_i, p, OP)
    assert r.sigma_e >= 0


# --------------------------------------------------------------------------
# The whole solve, against COM Octave's own MMSE.
#
# The assertions above check that the solve returns something of the right
# shape. None pins a value, and this function is the Rx FFE solve -- the place
# a numerical divergence would matter most and show least. Line coverage here
# was 43%.
#
# Expected values are COM Octave's MMSE, extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m by tools/octave_oracle.py, run on
# the pulse _mmse_inputs() rebuilds. Note the cursor convention: MATLAB indexes
# sbr from 1, SiCoPR from 0, so the oracle was given 33 and the port 32.
# --------------------------------------------------------------------------

_OCT_SIGMA_E = 0.0051408709476177131
_OCT_FOM = 36.236840869273557
_OCT_NW = 6
_OCT_C = [0.024779655353426432, -0.23660340681708794, 1.0,
          -0.34203316613344914, 0.1246626902634681, -0.022967639699143588]
_OCT_BLIM = [0.082293084489047597, 0.08952650178632432]

_M, _NUI, _CMX, _CPX, _NDFE, _CURSOR1, _FB = 4, 20, 2, 3, 2, 33, 100e9


def _mmse_inputs():
    """A pulse whose peak lands on a sampling-phase sample, plus a symmetric
    noise PSD. The noise is scaled so the solve sits at a realistic operating
    point (FOM about 36 dB) rather than in the noise-dominated regime, where
    the taps saturate and the comparison stops discriminating."""
    n = 80
    t = np.arange(n, dtype=float)
    sbr = 0.62 * np.exp(-((t - (_CURSOR1 - 1)) / 3.0) ** 2)
    sbr += 0.18 * np.exp(-((t - (_CURSOR1 - 1 + _M)) / 4.0) ** 2)
    sbr += 0.06 * np.exp(-((t - (_CURSOR1 - 1 - _M)) / 4.0) ** 2)
    sbr[t < _CURSOR1 - 1 - 3 * _M] = 0.0
    k = np.arange(64)
    S_n = 1e-15 * (0.85 ** np.minimum(k, 64 - k))
    return sbr, S_n


def _mmse_param():
    return SimpleNamespace(
        samples_per_ui=_M, num_ui_RXFF_noise=_NUI, levels=4, fb=_FB, R_LM=1,
        RxFFE_cmx=_CMX, RxFFE_cpx=_CPX, ndfe=_NDFE,
        bmax=np.array([0.85, 0.85]), bmin=np.array([-0.85, -0.85]),
        ffe_tapn_max=0.7, ffe_pre_tap1_max=0.7, ffe_post_tap1_max=0.7,
        ffe_pre_tap_len=_CMX, N_bg=0, N_bf=0, N_bmax=0, N_tail_start=1,
        bmaxg=0.2)


def _run_mmse():
    sbr, S_n = _mmse_inputs()
    return MMSE(SimpleNamespace(S_n=S_n), sbr, _CURSOR1 - 1, _mmse_param(),
                SimpleNamespace(RXFFE_FLOAT_CTL=0))


def test_matches_com_octave():
    r = _run_mmse()
    assert r.Nw == _OCT_NW, 'Nw is %s, COM Octave gives %s' % (r.Nw, _OCT_NW)
    for got, want, name in ((r.sigma_e, _OCT_SIGMA_E, 'sigma_e'),
                            (r.FOM, _OCT_FOM, 'FOM')):
        rel = abs(float(got) - want) / abs(want)
        assert rel < 1e-10, '%s is %.17g, COM Octave gives %.17g (rel %.2e)' % (
            name, float(got), want, rel)
    for name, got, want in (('C', np.asarray(r.C).ravel(), _OCT_C),
                            ('blim', np.asarray(r.blim).ravel(), _OCT_BLIM)):
        assert got.size == len(want), '%s has %d entries, expected %d' % (
            name, got.size, len(want))
        worst = float(np.max(np.abs(got - np.array(want))))
        assert worst < 1e-10, '%s worst difference from COM Octave is %.2e: %r' % (
            name, worst, list(got))


def test_main_tap_is_unity_and_taps_are_unclipped():
    """Guard the guard: the FFE is normalised so the main tap is 1, and no tap
    may sit on its limit, or the clipping branch is masking the solve."""
    r = _run_mmse()
    C = np.asarray(r.C).ravel()
    assert abs(C[_CMX] - 1.0) < 1e-12, 'main tap is %r, expected 1' % C[_CMX]
    off = np.delete(C, _CMX)
    assert np.all(np.abs(off) < 0.7 - 1e-9), (
        'a tap is sitting on ffe_tapn_max (%r), so the comparison is testing '
        'the clip and not the solve' % list(C))
    b = np.asarray(r.blim).ravel()
    assert np.all(np.abs(b) < 0.85 - 1e-9), (
        'a DFE tap is on its limit (%r)' % list(b))


# --------------------------------------------------------------------------
# MMSE carries its own copy of MMSE_FOM as _MMSE_FOM, so the copy has to be
# checked, not only the canonical. read_s4p_files is the cautionary tale: its
# inlined reader had drifted from read_p4_s4params on the skew binding and was
# 0.19 out while the canonical was right.
#
# Two things are pinned here. That the copy agrees with the canonical bit for
# bit on a well-conditioned problem, and that it takes the same decision the
# canonical does when the solve is singular -- ML 2645 is a backslash on a
# SQUARE system, where MATLAB warns and returns Inf, Octave returns a
# minimum-norm answer (a third behaviour), and the 2026-09-23 force() ruling
# has the port stop so the degenerate case stays visible.
# --------------------------------------------------------------------------

import scipy.linalg                                                # noqa: E402
from com_functions.fn.MMSE.py_impl import _MMSE_FOM                # noqa: E402
from com_functions.fn.MMSE_FOM.py_impl import MMSE_FOM as _canonical_MMSE_FOM  # noqa: E402


def _fom_case(singular=False):
    cmx, cpx, Nb, d = 2, 3, 2, 4
    Nw = cmx + 1 + cpx
    L = 4
    sigma_X2 = (L ** 2 - 1) / (3.0 * (L - 1) ** 2)
    h = np.array([0.02, 0.10, 0.62, 0.21, 0.07, 0.03, 0.01, 0.004])
    H = scipy.linalg.toeplitz(np.concatenate([h, np.zeros(Nw - 1)]),
                              np.concatenate([[h[0]], np.zeros(Nw - 1)]))
    rn = 0.02 ** 2 * (0.6 ** np.arange(Nw))
    Rnn = scipy.linalg.toeplitz(rn, rn)
    if singular:
        # a duplicate column makes R = H'H rank deficient, and no noise floor
        # leaves A EXACTLY singular rather than merely ill-conditioned
        H[:, 1] = H[:, 0]
        Rnn = np.zeros_like(Rnn)
    p = SimpleNamespace(RxFFE_cmx=cmx, RxFFE_cpx=cpx, N_bg=0, N_bf=0,
                        N_bmax=0, levels=L, R_LM=1,
                        bmax=np.full(Nb, 1.5), bmin=np.full(Nb, -1.5))
    return (p, H, Nb, Rnn, cmx, d, np.full(Nw, 50.0), np.full(Nw, -50.0),
            np.full(Nb, -1.5), np.full(Nb, 1.5), sigma_X2, None)


def test_inlined_MMSE_FOM_matches_the_canonical():
    args = _fom_case()
    a = _MMSE_FOM(*args)
    b = _canonical_MMSE_FOM(*args)
    assert float(a[0]) == float(b[0]), (
        'sigma_e: copy %.17g, canonical %.17g' % (a[0], b[0]))
    assert float(a[1]) == float(b[1]), (
        'FOM: copy %.17g, canonical %.17g' % (a[1], b[1]))
    assert int(a[4]) == int(b[4])
    for name, i in (('w', 2), ('blim', 5)):
        x, y = np.ravel(np.asarray(a[i])), np.ravel(np.asarray(b[i]))
        assert x.shape == y.shape, '%s shape %s vs %s' % (name, x.shape, y.shape)
        assert np.array_equal(x, y), (
            'the inlined copy has drifted: %s differs by up to %.3e'
            % (name, float(np.max(np.abs(x - y)))))


def test_inlined_singular_solve_stops_rather_than_guessing():
    args = _fom_case(singular=True)
    H = args[1]
    assert np.linalg.matrix_rank(H.T @ H) < H.shape[1], (
        'this fixture is not rank deficient any more, so the test cannot '
        'reach the singular branch')
    # the phrase has to name THIS solve: with lstsq substituted here the
    # call runs on and raises from the clipped-DFE solve instead
    with pytest.raises(ValueError, match=r'-Hb ib'):
        _MMSE_FOM(*args)


def test_inlined_singular_clipped_dfe_solve_stops():
    """h0 = 0 leaves A full rank and [R -h0'; h0 0] one short, which is
    the only way to reach ML 2669 with ML 2645 healthy."""
    args = list(_fom_case())
    H = np.array(args[1], dtype=float, copy=True)
    H[args[5], :] = 0.0          # args[5] is d
    args[1] = H
    with pytest.raises(ValueError, match=r'clipped-DFE'):
        with np.errstate(divide='ignore', invalid='ignore'):
            _MMSE_FOM(*args)


# ---------------------------------------------------------------------------
# ML 2609-2612: H = H(:, sel); HH = H'*H -- the Gram matrix of the SELECTED
# columns, formed on every call. A speed-up (3f1b7bb) hoisted H'*H of the full
# H out of the floating-tap search and gathered (H'*H)(sel,sel) from it. The two
# agree mathematically and sum in a different order, and no oracle can settle
# which last bit MATLAB lands on, so the reference's FORM is what is pinned:
# removed 2026-09-24 with the other speed-ups never verified against it.
# ---------------------------------------------------------------------------

def test_gram_matrix_is_formed_from_the_selected_columns():
    import inspect
    src = inspect.getsource(__import__('com_functions.fn.MMSE.py_impl', fromlist=['x']))
    assert 'HH_full' not in src, 'the hoisted Gram matrix is back'
    assert 'Hs = H.take(col_sel, 1)' in src and 'HH = Hs.T @ Hs' in src, (
        "HH must be H(:,sel)'*H(:,sel), as ML 2612 forms it")


# ---------------------------------------------------------------------------
# Ht (2026-09-24): the floating-tap search passes H.T laid out contiguously so
# each candidate gathers its columns as contiguous rows. Layout only: the
# Gram matrix is still H(:,sel)'*H(:,sel), and the result must be BIT-
# IDENTICAL with and without Ht, on every shape and selection.
# ---------------------------------------------------------------------------

def test_ht_layout_gives_bit_identical_results():
    rng = np.random.default_rng(21)
    for trial in range(40):
        Nw = int(rng.integers(18, 40))   # leaves room for cmx, cpx and a bank
        Nb = int(rng.integers(1, 4))
        cmx, cpx = 2, 5
        nrow = int(rng.integers(300, 1500))
        H = rng.standard_normal((nrow, Nw)) * 0.05
        H[80, :] += 1.0
        Rnn = np.eye(Nw) * 1e-3
        Nmax = Nw - cmx - cpx - 1
        param = SimpleNamespace(RxFFE_cmx=cmx, RxFFE_cpx=cpx, N_bmax=Nmax,
                                N_bf=1, N_bg=1, bmax=np.full(Nb, 0.85),
                                bmin=np.full(Nb, -0.85), R_LM=1.0, levels=4)
        k = int(rng.integers(1, max(2, Nmax - cpx)))
        idx = np.sort(rng.choice(np.arange(cpx + 1, Nmax + 1), size=min(k, Nmax - cpx),
                                 replace=False))
        args = (param, H, Nb, Rnn, cmx, 80, np.full(Nw, 10.0), np.full(Nw, -10.0),
                np.full(Nb, -0.85), np.full(Nb, 0.85), 1.0, idx.copy())
        a = _MMSE_FOM(*args)
        Ht = np.ascontiguousarray(H.T)
        b = _MMSE_FOM(*args[:-1], idx.copy(), Ht=Ht)
        c = _MMSE_FOM(*args[:-1], idx.copy(), Ht=Ht, G=Ht @ Ht.T)
        for r in (b, c):
            # Bit-identity between the layouts is a property of the BLAS kernel
            # the CPU selects, not of the code: it holds on the machine the
            # equivalence rule is judged on (tools/equivalence_check.py), and
            # CI runners, Windows and Linux alike, have shown 2e-14 relative.
            # Here it is held to that rule's own 1e-12 relative bound.
            np.testing.assert_allclose(a[1], r[1], rtol=1e-12, atol=0)
            np.testing.assert_allclose(a[2], r[2], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(a[0], r[0], rtol=1e-12, atol=1e-15)
