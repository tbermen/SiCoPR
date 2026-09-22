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
