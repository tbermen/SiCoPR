"""Tests for MMSE_FOM (MATLAB lines ~2580).

MATLAB GROUND TRUTH:
  Simple 2-tap FFE, 0 DFE:
    H = [[1, 0],[0, 1]]; Rnn = I; dw=0; d=0; wmax=[1,1]; wmin=[1,1]
    Expect: sigma_e finite, FOM finite, w shape (2,), blim shape (0,)
"""
import numpy as np
import pytest
from com_functions.fn.MMSE_FOM.py_impl import MMSE_FOM
from types import SimpleNamespace


def _param():
    p = SimpleNamespace()
    p.R_LM = 1.0
    p.levels = 2
    p.bmax = np.array([1.0, 1.0])
    p.bmin = np.array([-1.0, -1.0])
    p.RxFFE_cmx = 0
    p.RxFFE_cpx = 1
    p.N_bmax = 4
    return p


def test_basic_no_dfe():
    p = _param()
    H = np.eye(2)
    Nb = 0
    Rnn = np.eye(2)
    dw = 0
    d = 0
    wmax = np.ones(2)
    wmin = np.ones(2)
    bmin = np.array([])
    bmax = np.array([])
    sigma_X2 = 1.0
    sigma_e, FOM, w, idx, Nw, blim = MMSE_FOM(p, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2)
    assert np.isfinite(sigma_e), "sigma_e should be finite"
    assert np.isfinite(FOM), "FOM should be finite"
    assert len(w) == 2


def test_sigma_e_positive():
    p = _param()
    n = 4
    H = np.random.default_rng(42).standard_normal((n, n))
    Nb = 0
    Rnn = np.eye(n)
    dw = 1
    d = 1
    wmax = np.ones(n)
    wmin = -np.ones(n)
    wmax[dw] = 1.0
    wmin[dw] = 1.0
    bmin = np.array([])
    bmax = np.array([])
    sigma_X2 = 1.0 / 3.0
    sigma_e, FOM, w, idx, Nw, blim = MMSE_FOM(p, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2)
    assert sigma_e >= 0


def test_with_dfe():
    p = _param()
    n = 5
    H = np.eye(n)
    Nb = 2
    Rnn = np.eye(n)
    dw = 2
    d = 2
    wmax = np.ones(n)
    wmin = -np.ones(n)
    wmax[dw] = 1.0
    wmin[dw] = 1.0
    bmin = np.array([-1.0, -1.0])
    bmax = np.array([1.0, 1.0])
    p.bmax = bmax
    p.bmin = bmin
    sigma_X2 = 1.0 / 3.0
    sigma_e, FOM, w, idx, Nw, blim = MMSE_FOM(p, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2)
    assert sigma_e >= 0
    assert len(blim) == Nb


def test_fom_decreases_with_noise():
    p = _param()
    n = 3
    H = np.eye(n)
    Nb = 0
    bmin = np.array([])
    bmax = np.array([])
    dw = 1
    d = 1
    wmax = np.ones(n)
    wmin = -np.ones(n)
    wmax[dw] = 1.0
    wmin[dw] = 1.0
    sigma_X2 = 1.0 / 3.0

    _, FOM_low, _, _, _, _ = MMSE_FOM(p, H, Nb, np.eye(n), dw, d, wmax, wmin, bmin, bmax, sigma_X2)
    _, FOM_high, _, _, _, _ = MMSE_FOM(p, H, Nb, 10 * np.eye(n), dw, d, wmax, wmin, bmin, bmax, sigma_X2)
    assert FOM_low >= FOM_high


def test_output_shapes():
    p = _param()
    n = 4
    H = np.eye(n)
    Nb = 1
    Rnn = np.eye(n)
    dw = 1
    d = 1
    wmax = np.ones(n)
    wmin = -np.ones(n)
    wmax[dw] = 1.0
    wmin[dw] = 1.0
    bmin = np.array([-0.5])
    bmax = np.array([0.5])
    p.bmax = bmax
    p.bmin = bmin
    sigma_X2 = 1.0 / 3.0
    sigma_e, FOM, w, idx, Nw, blim = MMSE_FOM(p, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2)
    assert len(w) == n
    assert len(blim) == Nb
    assert isinstance(Nw, int)


# --------------------------------------------------------------------------
# Against COM Octave's own MMSE_FOM.
#
# Every assertion above this line checks a property -- isfinite, a length, a
# sign -- and not one pins a value, so they all pass on arithmetic that is
# simply wrong. This function had 100% line coverage and no value verification
# at all, which is the more dangerous of the two failure modes: the coverage
# number says it is exercised.
#
# Expected values are COM Octave's MMSE_FOM, extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m by tools/octave_oracle.py and run
# on the problem _mmse_case() builds.
# --------------------------------------------------------------------------

import scipy.linalg  # noqa: E402

_OCT_SIGMA_E = 0.028074115085696169
_OCT_FOM = 21.491453387493173
_OCT_NW = 6
_OCT_W = [-0.014103779473300368, -0.26701831323151454, 1.8175850688600781,
          -0.66633231905166135, -0.16042002231666302, 0.048413130609700268]
_OCT_BLIM = [-0.065621308281290938, -0.11546952014768284]


def _mmse_case():
    """A small but non-degenerate equalizer problem.

    d places the pulse peak on the main tap (index 1+dw). Getting that wrong
    makes the main tap negative, which inverts the wmax*w[dw] clip bounds and
    collapses every tap to the same value -- a solution that would pass a
    property test while telling you nothing.
    """
    cmx, cpx, Nb, d = 2, 3, 2, 4
    Nw = cmx + 1 + cpx
    L = 4
    sigma_X2 = (L ** 2 - 1) / (3.0 * (L - 1) ** 2)
    h = np.array([0.02, 0.10, 0.62, 0.21, 0.07, 0.03, 0.01, 0.004])
    H = scipy.linalg.toeplitz(np.concatenate([h, np.zeros(Nw - 1)]),
                              np.concatenate([[h[0]], np.zeros(Nw - 1)]))
    rn = 0.02 ** 2 * (0.6 ** np.arange(Nw))
    Rnn = scipy.linalg.toeplitz(rn, rn)
    wmax, wmin = np.full(Nw, 50.0), np.full(Nw, -50.0)
    bmax, bmin = np.full(Nb, 1.5), np.full(Nb, -1.5)
    p = SimpleNamespace(RxFFE_cmx=cmx, RxFFE_cpx=cpx, N_bg=0, N_bf=0,
                        N_bmax=0, levels=L, R_LM=1, bmax=bmax, bmin=bmin)
    return dict(param=p, H=H, Nb=Nb, Rnn=Rnn, dw=cmx, d=d, wmax=wmax,
                wmin=wmin, bmin=bmin, bmax=bmax, sigma_X2=sigma_X2)


def test_matches_com_octave():
    a = _mmse_case()
    sigma_e, FOM, w, _idx, Nw, blim = MMSE_FOM(
        a['param'], a['H'], a['Nb'], a['Rnn'], a['dw'], a['d'], a['wmax'],
        a['wmin'], a['bmin'], a['bmax'], a['sigma_X2'], None)

    assert Nw == _OCT_NW, 'Nw is %s, COM Octave gives %s' % (Nw, _OCT_NW)
    for got, want, name in ((sigma_e, _OCT_SIGMA_E, 'sigma_e'),
                            (FOM, _OCT_FOM, 'FOM')):
        rel = abs(float(got) - want) / abs(want)
        assert rel < 1e-11, '%s is %.17g, COM Octave gives %.17g (rel %.2e)' % (
            name, float(got), want, rel)
    for name, got, want in (('w', np.asarray(w).ravel(), _OCT_W),
                            ('blim', np.asarray(blim).ravel(), _OCT_BLIM)):
        assert got.size == len(want), '%s has %d entries, expected %d' % (
            name, got.size, len(want))
        worst = float(np.max(np.abs(got - np.array(want))))
        assert worst < 1e-11, '%s worst difference from COM Octave is %.2e: %r' % (
            name, worst, list(got))


def test_case_is_not_degenerate():
    """Guard the guard. If the taps ever come back all equal the problem has
    collapsed into the clipped regime and the comparison above is vacuous."""
    a = _mmse_case()
    _s, _f, w, _i, _n, blim = MMSE_FOM(
        a['param'], a['H'], a['Nb'], a['Rnn'], a['dw'], a['d'], a['wmax'],
        a['wmin'], a['bmin'], a['bmax'], a['sigma_X2'], None)
    w = np.asarray(w).ravel()
    assert np.ptp(w) > 0.5, (
        'the equalizer taps span only %.3g; a degenerate solve makes this test '
        'prove nothing' % np.ptp(w))
    b = np.asarray(blim).ravel()
    assert np.all(np.abs(b) < 1.5 - 1e-9), (
        'the DFE taps are sitting on their limit (%r), so the clipping branch '
        'is masking the solve' % list(b))


# --------------------------------------------------------------------------
# The MMSE solve when its matrix is singular.
#
# ML 2645 (Z = A\Ct) and ML 2669 (wl = [R -h0'; h0 0]\[...]) are backslashes
# on SQUARE systems, and the three languages disagree on exactly one input:
#
#   MATLAB   warns, returns Inf, and the NaNs that follow make the candidate
#            lose. Measured under Octave with octave/patches/mldivide_matlab.m
#            restoring MATLAB semantics: sigma_e NaN, FOM NaN, w all NaN.
#   Octave   returns a minimum-norm least-squares answer instead: a finite
#            sigma_e of 0.0904 and a FOM of 11.34 on this very case, which is
#            a different answer, not a rounding difference.
#   numpy    np.linalg.solve raises; np.linalg.lstsq would give Octave's.
#
# The 2026-09-23 ruling on force() settled which of those the port takes: a
# silent minimum-norm answer is the one outcome NEITHER reference produces, so
# the port stops and the degenerate case stays visible. The divergence from
# the reference's NaN-and-continue is recorded in tests/test_optimizer_mmse.py.
#
# This test is also what stops the solve silently becoming an lstsq: lstsq
# returns rather than raising, so swapping it makes this fail.
# --------------------------------------------------------------------------

def _singular_mmse_case():
    """A rank-deficient MMSE problem.

    Two identical columns in H make R = H'H rank deficient, and Rnn = 0 leaves
    no noise floor to regularise it, so A is EXACTLY singular rather than
    merely ill-conditioned. That distinction is the one that matters: MATLAB
    returns a finite answer for an ill-conditioned matrix and Inf only for an
    exactly singular one.
    """
    a = _mmse_case()
    H = np.array(a['H'], dtype=float, copy=True)
    H[:, 1] = H[:, 0]
    a['H'] = H
    a['Rnn'] = np.zeros_like(np.asarray(a['Rnn'], dtype=float))
    return a


def test_singular_mmse_solve_stops_rather_than_guessing():
    a = _singular_mmse_case()
    A_is_singular = np.linalg.matrix_rank(a['H'].T @ a['H']) < a['H'].shape[1]
    assert A_is_singular, (
        'this fixture is not rank deficient any more, so the test cannot '
        'reach the singular branch')

    # the phrase has to name THIS solve: with lstsq substituted here the
    # call runs on and raises from the clipped-DFE solve instead, and a
    # looser match would pass on the mutant
    with pytest.raises(ValueError, match=r'-Hb ib'):
        MMSE_FOM(a['param'], a['H'], a['Nb'], a['Rnn'], a['dw'], a['d'],
                 a['wmax'], a['wmin'], a['bmin'], a['bmax'], a['sigma_X2'],
                 None)


def _zero_h0_case():
    """A healthy A with a singular clipped-DFE system behind it.

    h0 is H[d], so zeroing row d makes h0 = 0. For a positive-definite R
    the bordered matrix [R -h0'; h0 0] is singular exactly when
    h0' inv(R) h0 is zero, so A stays full rank (8 of 8) while Rb loses
    one (6 of 7). That is the only way to reach ML 2669 with ML 2645
    healthy.
    """
    a = _mmse_case()
    H = np.array(a['H'], dtype=float, copy=True)
    H[a['d'], :] = 0.0
    a['H'] = H
    return a


def test_singular_clipped_dfe_solve_stops_rather_than_guessing():
    a = _zero_h0_case()
    with pytest.raises(ValueError, match=r'clipped-DFE'):
        with np.errstate(divide='ignore', invalid='ignore'):
            MMSE_FOM(a['param'], a['H'], a['Nb'], a['Rnn'], a['dw'],
                     a['d'], a['wmax'], a['wmin'], a['bmin'], a['bmax'],
                     a['sigma_X2'], None)


def test_singular_case_would_otherwise_return_octaves_answer():
    """Guard the guard: show the case really does separate solve from lstsq.

    If lstsq failed on this input too, the test above would pass for the wrong
    reason. It does not: lstsq returns the minimum-norm answer, which is what
    Octave's backslash gives and what the port must not silently adopt.
    """
    a = _singular_mmse_case()
    R = a['H'].T @ a['H'] + np.asarray(a['Rnn'], dtype=float)
    rhs = np.zeros(R.shape[0])
    rhs[a['dw']] = 1.0
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(R, rhs)
    x, _res, _rank, _sv = np.linalg.lstsq(R, rhs, rcond=None)
    assert np.all(np.isfinite(x)), (
        'lstsq also fails here, so this input does not separate the two')


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
    src = inspect.getsource(__import__('com_functions.fn.MMSE_FOM.py_impl', fromlist=['x']))
    assert 'HH_full' not in src, 'the hoisted Gram matrix is back'
    assert 'Hs = H.take(col_sel, 1)' in src and 'HH = Hs.T @ Hs' in src, (
        "HH must be H(:,sel)'*H(:,sel), as ML 2612 forms it")


def test_mmse_fom_takes_no_precomputed_gram_matrix():
    import inspect
    assert 'HH_full' not in inspect.signature(MMSE_FOM).parameters


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
        a = MMSE_FOM(*args)
        Ht = np.ascontiguousarray(H.T)
        b = MMSE_FOM(*args[:-1], idx.copy(), Ht=Ht)
        c = MMSE_FOM(*args[:-1], idx.copy(), Ht=Ht, G=Ht @ Ht.T)
        for r in (b, c):
            # Bit-identity between the layouts is a property of the BLAS kernel
            # the CPU selects, not of the code: it holds on the machine the
            # equivalence rule is judged on (tools/equivalence_check.py), and
            # CI runners, Windows and Linux alike, have shown 2e-14 relative.
            # Here it is held to that rule's own 1e-12 relative bound.
            np.testing.assert_allclose(a[1], r[1], rtol=1e-12, atol=0)
            np.testing.assert_allclose(a[2], r[2], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(a[0], r[0], rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 4p17p0: HH from H(:,1)'*H by lag (ML 2640-2702 of com_ieee8023_4p17p0.m).
#
# COM Octave values: MMSE_FOM extracted verbatim from
# octave/com_ieee8023_4p17p0_octave_compat.m by tools/octave_oracle.py, run on
# the two problems _case_417 builds, with HH_val = H(:,1)'*H formed in Python
# and passed to both sides. Generator: runs/4p17p0_oracles/gen_mmse_fom_oracle.py
# (local). Residual ~1e-13 relative: the order the linear solve accumulates in.
#
# FULL: complete Toeplitz H, no floating taps; the lookup equals H'*H in exact
# arithmetic. TRUNC: an H whose columns are cut short across NONZERO samples,
# with floating taps. There H(:,1)'*H by lag is not H'*H, so the two releases
# answer differently, which is what lets this test tell the forms apart. MMSE
# itself never builds such an H (its truncated branch drops only zero padding,
# see MMSE's 4p17p0 test), so in the pipeline the forms agree to rounding; this
# case exists to pin which form is followed. The 4p16p0 value, run through
# 4p16p0's MMSE_FOM, is the negative control: a 4p17p0 path that quietly formed
# H'*H would give it.
# ---------------------------------------------------------------------------

_OCT417 = {
    'FULL': (0.05177049124753655, 16.175929185617246,
             [-0.00917767010494733, -0.27014445600796705, 1.6558649645831518,
              0.3036460917637337, 0.018594272554600094, -0.10565191264956038],
             [0.5165531664347394, 0.17271870423196065]),
    'TRUNC': (0.05090976631086431, 16.3215528352725,
              [-0.00888530495712958, -0.2715327783670887, 1.6541988350219803,
               0.3508630606139819, -0.1522864485469117, -0.05206521770225447,
               -0.0204208702030805, 0.008078953610315636],
              [0.5293730500921382, 0.08188276405929407]),
}
_OCT416_TRUNC_FOM = 16.881615193307734


def _case_417(truncated):
    cmx, cpx, Nb, L = 2, 3, 2, 4
    sigma_X2 = (L ** 2 - 1) / (3.0 * (L - 1) ** 2)
    h = np.array([0.02, 0.10, 0.62, 0.21, 0.07, 0.03, -0.02, 0.012, 0.006,
                  -0.004, 0.003, 0.002, 0.0015, 0.001])
    if truncated:
        Nmax, N_bf, N_bg = 9, 1, 2
        Nw = cmx + Nmax + 1
        H = scipy.linalg.toeplitz(h[:10], np.concatenate([[h[0]], np.zeros(Nw - 1)]))
        idx = np.array([6, 8])
        wmax = np.concatenate([np.full(cmx + 1 + cpx, 50.0), np.full(N_bf * N_bg, 0.2)])
    else:
        Nmax, N_bf, N_bg = 0, 0, 0
        Nw = cmx + 1 + cpx
        H = scipy.linalg.toeplitz(np.concatenate([h, np.zeros(Nw - 1)]),
                                  np.concatenate([[h[0]], np.zeros(Nw - 1)]))
        idx = np.array([], dtype=int)
        wmax = np.full(Nw, 50.0)
    rn = 0.02 ** 2 * (0.6 ** np.arange(Nw))
    bmax, bmin = np.full(Nb, 1.5), np.full(Nb, -1.5)
    p = SimpleNamespace(RxFFE_cmx=cmx, RxFFE_cpx=cpx, N_bg=N_bg, N_bf=N_bf,
                        N_bmax=Nmax, levels=L, R_LM=1, bmax=bmax, bmin=bmin)
    return (p, H, Nb, scipy.linalg.toeplitz(rn, rn), cmx, cmx + 2, wmax, -wmax,
            bmin, bmax, sigma_X2, idx)


@pytest.mark.parametrize('name', ['FULL', 'TRUNC'])
def test_4p17p0_hh_lookup_matches_com_octave(name):
    args = _case_417(name == 'TRUNC')
    H = args[1]
    sigma_e, FOM, w, _idx, _Nw, blim = MMSE_FOM(*args, HH_val=H[:, 0] @ H)
    want_s, want_F, want_w, want_b = _OCT417[name]
    assert abs(sigma_e - want_s) / want_s < 1e-11
    assert abs(FOM - want_F) / want_F < 1e-11
    np.testing.assert_allclose(w, want_w, rtol=0, atol=1e-11)
    np.testing.assert_allclose(blim, want_b, rtol=0, atol=1e-11)


def test_4p17p0_truncated_h_is_not_the_4p16p0_answer():
    args = _case_417(True)
    H = args[1]
    FOM_417 = MMSE_FOM(*args, HH_val=H[:, 0] @ H)[1]
    FOM_416 = MMSE_FOM(*args)[1]
    assert abs(FOM_416 - _OCT416_TRUNC_FOM) / _OCT416_TRUNC_FOM < 1e-11
    assert abs(FOM_417 - FOM_416) > 0.5, (
        'the lag lookup and H\'*H agree on a truncated H: the 4p17p0 path is '
        'not doing what the reference does')
