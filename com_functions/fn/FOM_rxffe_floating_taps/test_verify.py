"""Verification tests for FOM_rxffe_floating_taps().

# ============================================================
# MATLAB GROUND TRUTH (lines 2063-2108)
# Greedy floating-tap bank placement: choose N_bg banks (N_bf taps each), one bank
# at a time, each at the location maximising MMSE_FOM over the candidate tap set.
# Returns sorted idx = all_idx + RxFFE_cpx (1-based, untrimmed-hisi positions).
# ============================================================
"""
import numpy as np
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.FOM_rxffe_floating_taps.py_impl import FOM_rxffe_floating_taps


# ---- 1. Greedy-selection logic (stub kernel) ------------------------------
def test_greedy_picks_highest_isi_banks():
    """With FOM = -(uncovered ISI energy), greedy must cover the largest banks."""
    RxFFE_cpx = 1
    isi = np.array([0.02, 0.40, 0.35, 0.03, 0.05, 0.30, 0.28, 0.02])

    def stub(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
                 **_kw):
        covered = set(int(i) - RxFFE_cpx - 1 for i in np.atleast_1d(idx))
        resid = sum(isi[j] ** 2 for j in range(len(isi)) if j not in covered)
        return (0.0, -resid, None, idx, 0, None)

    param = SimpleNamespace(RxFFE_cpx=RxFFE_cpx, N_bmax=len(isi) + RxFFE_cpx, N_bf=2, N_bg=2)
    h = np.concatenate([np.zeros(RxFFE_cpx), isi])
    idx = FOM_rxffe_floating_taps(param, h, None, 1, None, 0, 0, None, None, None, None,
                                  1.0, 0, RxFFE_cpx + len(isi), _MMSE_FOM_fn=stub)
    covered0 = sorted(int(i) - RxFFE_cpx - 1 for i in idx)
    assert covered0 == [1, 2, 5, 6], covered0


def test_idx_convention_and_no_overlap():
    """idx is sorted, count == N_bg*N_bf, values are untrimmed-hisi 1-based positions."""
    RxFFE_cpx = 2
    isi = np.array([0.5, 0.1, 0.05, 0.4, 0.05, 0.05, 0.3, 0.02])

    def stub(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
                 **_kw):
        covered = set(int(i) - RxFFE_cpx - 1 for i in np.atleast_1d(idx))
        return (0.0, -sum(isi[j] ** 2 for j in range(len(isi)) if j not in covered), None, idx, 0, None)

    N_bf, N_bg = 1, 3
    param = SimpleNamespace(RxFFE_cpx=RxFFE_cpx, N_bmax=len(isi) + RxFFE_cpx, N_bf=N_bf, N_bg=N_bg)
    h = np.concatenate([np.zeros(RxFFE_cpx), isi])
    idx = FOM_rxffe_floating_taps(param, h, None, 1, None, 0, 0, None, None, None, None,
                                  1.0, 0, RxFFE_cpx + len(isi), _MMSE_FOM_fn=stub)
    assert list(idx) == sorted(idx)
    assert len(idx) == N_bg * N_bf
    assert all(RxFFE_cpx + 1 <= v <= RxFFE_cpx + len(isi) for v in idx)
    # single-tap banks pick the three largest ISI: positions 0,3,6 -> +RxFFE_cpx+1
    assert sorted(int(i) - RxFFE_cpx - 1 for i in idx) == [0, 3, 6]


# ---- 2. Integration with the assembled real MMSE_FOM ----------------------
def test_integration_real_mmse_fom():
    """End-to-end: real sicopr.MMSE_FOM selects valid floating taps and accepts them."""
    import importlib
    sicopr = importlib.import_module('sicopr')

    RxFFE_cmx, RxFFE_cpx = 2, 2
    N_bf, N_bg, N_bmax, ndfe = 1, 2, 6, 1
    dw = RxFFE_cmx
    Nw = dw + N_bmax + 1            # 9
    dh = 2
    d = dw + dh                    # 4
    param = SimpleNamespace(
        RxFFE_cmx=RxFFE_cmx, RxFFE_cpx=RxFFE_cpx, N_bf=N_bf, N_bg=N_bg, N_bmax=N_bmax,
        ndfe=ndfe, R_LM=1.0, levels=4, bmax=np.array([0.7]), bmin=np.array([-0.7]),
    )
    sigma_X2 = (param.levels ** 2 - 1) / (3 * (param.levels - 1) ** 2)

    h = np.array([0.05, 0.1, 1.0, 0.3, 0.05, 0.4, 0.35, 0.05, 0.3, 0.02, 0, 0, 0, 0, 0, 0])
    from scipy.linalg import toeplitz
    hc1 = np.concatenate([h, np.zeros(Nw - 1)])
    hr1 = np.concatenate([[h[0]], np.zeros(Nw - 1)])
    H = toeplitz(hc1, hr1)
    Rn = np.exp(-np.arange(Nw))    # decaying -> PD Toeplitz
    Rnn = toeplitz(Rn[:Nw], Rn[:Nw])
    wmax = np.ones(20) * 10.0
    wmin = -np.ones(20) * 10.0
    isi_start, isi_end = dh + 1, (dh - dw) + Nw   # 3, 9

    idx = sicopr.FOM_rxffe_floating_taps(param, h, H, ndfe, Rnn, dw, d, wmax, wmin,
                                      param.bmin, param.bmax, sigma_X2, isi_start, isi_end)
    # valid: sorted, right count, columns within H
    assert list(idx) == sorted(idx) and len(idx) == N_bg * N_bf
    assert all(0 <= v + RxFFE_cmx < H.shape[1] for v in idx)
    # the real kernel must produce a finite FOM for the selected taps
    sigma_e, FOM, w, idx_out, Nw_out, blim = sicopr.MMSE_FOM(
        param, H, ndfe, Rnn, dw, d, wmax, wmin, param.bmin, param.bmax, sigma_X2, idx)
    assert np.isfinite(FOM), FOM
    # the biggest ISI taps (h indices 5,6 -> trimmed positions 0,1) should be chosen
    chosen_trimmed = sorted(int(v) - RxFFE_cpx - 1 for v in idx)
    assert chosen_trimmed[0] == 0, chosen_trimmed


if __name__ == '__main__':
    test_greedy_picks_highest_isi_banks()
    test_idx_convention_and_no_overlap()
    test_integration_real_mmse_fom()
    print('FOM_rxffe_floating_taps: ALL TESTS PASSED')


# --------------------------------------------------------------------------
# The bank search, against COM Octave's own FOM_rxffe_floating_taps.
#
# None of the assertions above pins a value: they check lengths, sortedness
# and membership, all of which hold for a search that picks the wrong taps.
# This is the floating-tap search, so picking the wrong taps is the failure
# that matters.
#
# The expected locations are COM Octave's, via tools/octave_oracle.py, on the
# ISI vector _isi_case() rebuilds. That vector carries deliberate bumps, and
# test_search_follows_the_isi_bumps checks the answer actually tracks them --
# a search that returned the first N locations regardless would otherwise sail
# through.
# --------------------------------------------------------------------------

import scipy.linalg  # noqa: E402

from com_functions.fn.MMSE_FOM.py_impl import MMSE_FOM as _MMSE_FOM  # noqa: E402

_F_CMX, _F_CPX, _F_NB = 2, 3, 2
_F_NBG, _F_NBF, _F_NBMAX = 2, 2, 12
_F_NCOL, _F_D, _F_L = 16, 5, 4
_F_SIGMA_X2 = (_F_L ** 2 - 1) / (3.0 * (_F_L - 1) ** 2)
_OCT_IDX = [5, 6, 8, 9]


def _isi_case(bumps=((9, 0.16), (10, 0.12), (17, 0.10))):
    rng = np.random.default_rng(7)
    n = 24
    k = np.arange(n, dtype=float)
    h = 0.35 * np.exp(-k / 5.0) + 0.02 * rng.standard_normal(n)
    for i, a in bumps:
        h[i] += a
    H = scipy.linalg.toeplitz(
        np.concatenate([h, np.zeros(_F_NCOL - 1)]),
        np.concatenate([[h[0]], np.zeros(_F_NCOL - 1)]))
    rn = 0.01 ** 2 * (0.5 ** np.arange(_F_NCOL))
    Rnn = scipy.linalg.toeplitz(rn, rn)
    p = SimpleNamespace(RxFFE_cmx=_F_CMX, RxFFE_cpx=_F_CPX, N_bg=_F_NBG,
                        N_bf=_F_NBF, N_bmax=_F_NBMAX, levels=_F_L, R_LM=1,
                        bmax=np.full(_F_NB, 1.5), bmin=np.full(_F_NB, -1.5))
    return dict(param=p, h=h, H=H, Nb=_F_NB, Rnn=Rnn, dw=_F_CMX, d=_F_D,
                wmax=np.full(_F_NCOL, 50.0), wmin=np.full(_F_NCOL, -50.0),
                bmin=np.full(_F_NB, -1.5), bmax=np.full(_F_NB, 1.5),
                sigma_X2=_F_SIGMA_X2, isi_start=0, isi_end=n)


def _search(a):
    return np.asarray(FOM_rxffe_floating_taps(
        a['param'], a['h'], a['H'], a['Nb'], a['Rnn'], a['dw'], a['d'],
        a['wmax'], a['wmin'], a['bmin'], a['bmax'], a['sigma_X2'],
        a['isi_start'], a['isi_end'], _MMSE_FOM_fn=_MMSE_FOM)).ravel()


def test_bank_search_matches_com_octave():
    got = _search(_isi_case())
    assert list(got) == _OCT_IDX, (
        'the search chose taps %r; COM Octave chooses %r on the same ISI'
        % (list(got), _OCT_IDX))


def test_search_follows_the_isi_bumps():
    """Move the ISI energy and the chosen taps must move with it, or the
    comparison above is pinning a search that ignores its input."""
    moved = _search(_isi_case(bumps=((4, 0.16), (5, 0.12), (17, 0.10))))
    assert list(moved) != _OCT_IDX, (
        'the search returned %r for a different ISI profile too, so it is not '
        'responding to the input it is given' % list(moved))


def test_returns_two_banks_of_the_configured_size():
    got = _search(_isi_case())
    assert got.size == _F_NBG * _F_NBF, (
        '%d taps for %d groups of %d' % (got.size, _F_NBG, _F_NBF))
    assert list(got) == sorted(got), 'locations must come back sorted: %r' % list(got)


def test_each_candidate_is_scored_from_h_itself():
    """ML 2083: every candidate calls MMSE_FOM(param,H,...,idx), which forms
    H(:,sel)'*H(:,sel) itself. The port once passed a precomputed Gram matrix
    of the full H (HH_full) for speed; removed 2026-09-24 as never verified
    against the reference. The kernel may receive H.T laid out contiguously
    (Ht, layout only) and G = Ht @ Ht.T -- the verified form of that hoist,
    whose gathered block is bit-identical to forming it per candidate (pinned
    in MMSE_FOM/test_verify.py) -- and nothing else."""
    seen = []
    isi = np.array([0.02, 0.40, 0.35, 0.03, 0.05, 0.30])

    def spy(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
            **kw):
        seen.append(kw)
        return (0.0, float(np.sum(np.atleast_1d(idx))), None, idx, 0, None)

    param = SimpleNamespace(RxFFE_cpx=1, N_bmax=len(isi) + 1, N_bf=2, N_bg=2)
    h = np.concatenate([np.zeros(1), isi])
    H = np.eye(8)
    FOM_rxffe_floating_taps(param, h, H, 1, None, 0, 0, None, None, None, None,
                            1.0, 0, 1 + len(isi), _MMSE_FOM_fn=spy)
    assert seen
    for kw in seen:
        assert set(kw) <= {'Ht', 'G'}, kw
        if 'Ht' in kw:
            np.testing.assert_array_equal(kw['Ht'], H.T)
            assert kw['Ht'].flags['C_CONTIGUOUS']
        if 'G' in kw:
            np.testing.assert_array_equal(kw['G'], kw['Ht'] @ kw['Ht'].T)
