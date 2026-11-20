"""Verification tests for fast_clip_topK (new in 4p18p0, Clip Method = Fast_1).

COM Octave: fast_clip_topK extracted verbatim from
octave/com_ieee8023_4p18p0_octave_compat.m by tools/octave_oracle.py and run on
the CASES below (a 200-sample synthetic pulse at PAM2/4/6, a pulse shorter than
K, and tied magnitudes). Generator: runs/4p18p0_oracles/gen_fast_clip_oracle.py
(local).

Equivalent mutants (shown to survive, 2026-11-19), so no test can pin them:
the order of tied magnitudes in the sort, and the column-major flatten of the
state expansion. The PAM levels are symmetric, so S_top's distribution depends
on |h_k| alone, and the states enter only through a sum compared against P_qc,
never through their order. Caught: the colon range for odd L, returning t_lo,
and ctle_signal_sigma formed from the enumerated states.

Residual, named and bounded: adc_clip is the end of a bisection that starts at
t_hi = sum(abs(h)) + 10*sigma_rest. numpy sums pairwise, Octave strictly left to
right (MATLAB SIMD-blocked, a third order), so on the 200-sample pulse t_hi
starts 1 ulp apart and adc_clip ends 1-2 ulp apart. With a left-to-right sum
the port returns COM Octave's adc_clip exactly (test below), so every bisection
decision is the same. ctle_signal_sigma is bit-exact in every case.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

import com_functions.fn.fast_clip_topK.py_impl as _mod  # noqa: E402
from com_functions.fn.fast_clip_topK.py_impl import fast_clip_topK  # noqa: E402


def _pulse(n=200, seed=3):
    rng = np.random.RandomState(seed)
    h = np.zeros(n)
    h[:6] = [0.01, -0.04, 0.12, 0.55, 0.21, -0.06]
    h[6:] = rng.randn(n - 6) * 0.02 * np.exp(-np.arange(n - 6) / 30.0)
    return h


_CASES = {
    # name: (h, L, sigma_noise, P_qc, BinSize, K)
    'pam4_k6': (_pulse(), 4, 0.012, 1e-4, 1e-5, 6),
    'pam2_k12': (_pulse(), 2, 0.012, 1e-4, 1e-5, 12),
    'pam6_k5': (_pulse(), 6, 0.008, 1e-3, 1e-5, 5),
    'k_gt_n': (np.array([0.5, -0.2, 0.1, 0.05]), 4, 0.01, 1e-4, 1e-6, 6),
    'ties': (np.array([0.3, -0.3, 0.1, 0.3, -0.1, 0.02, 0.1]), 4, 0.02, 1e-3, 1e-5, 3),
}

# COM Octave, fast_clip_topK on _CASES: (adc_clip, ctle_signal_sigma)
_OCT = {
    'pam4_k6': (1.0770338718762589, 0.45509783822440875),
    'pam2_k12': (1.1926807832136253, 0.6104834774518916),
    'pam6_k5': (0.9580696495936898, 0.41703577254173924),
    'k_gt_n': (0.8694982051849363, 0.4100677450806825),
    'ties': (1.1035915140462778, 0.4090096440047458),
}
_SHORT = ('k_gt_n', 'ties')   # fewer than 8 terms: every sum order is the same


def _sx2(L):
    return (L ** 2 - 1) / (3 * (L - 1) ** 2)


def _run(name):
    h, L, sn, pqc, bs, K = _CASES[name]
    return fast_clip_topK(h, L, _sx2(L), sn, pqc, SimpleNamespace(BinSize=bs), K)


@pytest.mark.parametrize('name', sorted(_OCT))
def test_matches_com_octave(name):
    a, s = _run(name)
    assert s == _OCT[name][1]
    if name in _SHORT:
        assert a == _OCT[name][0]
    else:
        assert abs(a - _OCT[name][0]) <= 4 * np.spacing(_OCT[name][0])


class _LeftToRight:
    """numpy, but with Octave's left-to-right sum()."""

    def __getattr__(self, k):
        return getattr(np, k)

    @staticmethod
    def sum(x):
        acc = 0.0
        for v in np.asarray(x, dtype=float).ravel():
            acc += v
        return acc


def test_with_octaves_sum_order_adc_clip_is_exact(monkeypatch):
    # The residual above is ONLY the order of accumulation: summed as Octave
    # sums, every case is bit-exact.
    monkeypatch.setattr(_mod, 'np', _LeftToRight())
    for name, (want_a, want_s) in _OCT.items():
        a, s = _run(name)
        assert (a, s) == (want_a, want_s), name


def test_odd_pam_order_errors_as_the_reference_does():
    # COM Octave, L=3, K=8: "product: nonconformant arguments
    # (op1 is 1x6561, op2 is 1x256)" -- 1:L/2 gives two levels, ones(1,L)/L three.
    h = _pulse()
    with pytest.raises(ValueError, match=r'nonconformant arguments \(op1 is 1x6561, op2 is 1x256\)'):
        fast_clip_topK(h, 3, _sx2(3), 0.012, 1e-4, SimpleNamespace(BinSize=1e-5), 8)
