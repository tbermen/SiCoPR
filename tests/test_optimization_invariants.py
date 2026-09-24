"""Pin the invariants the August-2026 speed work depends on.

The engine was optimised from 16.1 h to ~3.5 h per 208-case corpus on the
explicit condition that no reported value changed. Three of the four changes
are only safe because of an invariant that is easy to break later and produces
NO error when broken -- just quietly wrong numbers:

  1. Memoised ADC-clip PDF (get_PSDs). Correct only while the cache key covers
     everything the result depends on, and while callers cannot mutate a cached
     object through the reference they are handed.
  2. Hoisted Gram matrix (MMSE / MMSE_FOM) -- REMOVED 2026-09-24. It summed
     in a different order from the reference's H(:,sel)'*H(:,sel).
  3. Size-gated FFT convolution -- REMOVED 2026-09-23 (8ec85b0). It buried the
     far tail of every PDF in round-off.
  4. Shared np.eye / np.zeros caches in the MMSE block assembly. Correct only
     while nobody writes into the shared arrays.

None of these is checked by the per-function tests, which exercise one function
with one set of inputs and never call it twice.

Run: python tests/test_optimization_invariants.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import copy
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402
import sicopr  # noqa: E402


def _load(fn_name):
    """Import one py_impl module directly, to reach its module-level caches."""
    path = os.path.join(_ROOT, 'com_functions', 'fn', fn_name, 'py_impl.py')
    spec = importlib.util.spec_from_file_location('_pi_' + fn_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =========================================================== 1. PDF memoisation
psd = _load('get_PSDs')
rng = np.random.default_rng(7)
vec = rng.standard_normal(400) * 1e-3

psd._PDF_CACHE.clear()
cold = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)
warm = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)

check("pdf_cache_hit_is_bit_identical",
      np.array_equal(np.asarray(cold.y, dtype=float),
                     np.asarray(warm.y, dtype=float))
      and np.array_equal(np.asarray(cold.x, dtype=float),
                         np.asarray(warm.x, dtype=float)),
      "a cache hit returned a different PDF than the cold computation -- the "
      "memoisation is not transparent")

check("pdf_cache_actually_hit",
      len(psd._PDF_CACHE) == 1,
      "expected one cache entry after two identical calls, found %d; the key "
      "is over-specified and the cache is doing no work"
      % len(psd._PDF_CACHE))

# A different input must NOT collide.
other = psd._get_pdf_from_sampled_signal(vec * 2.0, 4, 1e-5)
check("pdf_cache_distinguishes_inputs",
      len(psd._PDF_CACHE) == 2
      and not np.array_equal(np.asarray(other.y, dtype=float),
                             np.asarray(cold.y, dtype=float)),
      "a different input vector returned the cached result -- key collision")

# Parameters other than the vector must participate in the key.
psd._PDF_CACHE.clear()
a = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)
b = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 2e-5)
check("pdf_cache_key_includes_binsize",
      len(psd._PDF_CACHE) == 2,
      "BinSize is not part of the cache key: two different bin sizes produced "
      "one entry, so one of them got the other's PDF")

# The returned object must not be a window onto the cache.
psd._PDF_CACHE.clear()
first = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)
baseline = np.asarray(first.y, dtype=float).copy()
first.y = np.asarray(first.y, dtype=float) * 3.0      # rebind, the documented use
second = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)
check("pdf_cache_survives_caller_rebinding_a_field",
      np.array_equal(np.asarray(second.y, dtype=float), baseline),
      "rebinding a field on a returned PDF corrupted the cached entry")

third = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)
np.asarray(third.y, dtype=float)[:] *= 3.0            # IN-PLACE: the hazard
fourth = psd._get_pdf_from_sampled_signal(vec.copy(), 4, 1e-5)
check("pdf_cache_survives_caller_mutating_in_place",
      np.array_equal(np.asarray(fourth.y, dtype=float), baseline),
      "an IN-PLACE write through a returned PDF corrupted the cache, so every "
      "later hit is wrong. The cache returns SimpleNamespace(**vars(hit)), "
      "which copies the namespace but SHARES the arrays -- safety currently "
      "rests on callers rebinding (pdf.y = pdf.y * k) rather than mutating "
      "(pdf.y *= k). Either copy the arrays on the way out or keep every "
      "caller to rebinding.")

# =========================================================== 2. Gram hoist, removed
# 2026-09-24: HH is H(:,sel)'*H(:,sel) per call again, as ML 2612 forms it; the
# hoisted full Gram matrix summed in a different order and was never verified
# against the reference. The fixture below still drives sections 4-5.
mm = _load('MMSE_FOM')
rng = np.random.default_rng(11)
Nw, Nb, num = 12, 3, 200
H = rng.standard_normal((num, Nw)) * 0.1
H[40, :] += 1.0
Rnn = np.eye(Nw) * 1e-3
param = sicopr.SimpleNamespace(RxFFE_cmx=4, RxFFE_cpx=7, N_bmax=Nb, N_bf=0,
                            N_bg=0, bmax=np.full(Nb, 0.85),
                            bmin=np.full(Nb, -0.85), R_LM=1.0, levels=4)
kw = dict(param=param, H=H, Nb=Nb, Rnn=Rnn, dw=4, d=40,
          wmax=np.full(Nw, 10.0), wmin=np.full(Nw, -10.0),
          bmin=np.full(Nb, -0.85), bmax=np.full(Nb, 0.85), sigma_X2=1.0)

r_self = mm.MMSE_FOM(**kw, idx=None)
check('gram_hoist_is_gone',
      'HH_full' not in __import__('inspect').signature(mm.MMSE_FOM).parameters,
      'MMSE_FOM takes a precomputed Gram matrix again')

# =========================================================== 3. conv is direct
# _conv1d lives with the canonical conv_fct. It used to dispatch to an FFT
# above 128 bins; that put round-off of ~eps*peak into every tail bin of the
# noise CDF and was removed on 2026-09-23 (8ec85b0), so there is no gate left
# to check. What remains is stronger: _conv1d IS np.convolve, bit for bit, at
# every size, including those the old gate sent to the FFT.
cf = _load('conv_fct')
check("conv1d_has_no_fft_gate", not hasattr(cf, '_CONV_FFT_MIN'),
      "conv_fct grew a size gate again; an FFT path loses the far tail of "
      "every PDF it touches (see conv_fct/test_verify.py tail test)")
for n in (8, 64, 127, 128, 129, 512):
    a = rng.standard_normal(n)
    b = rng.standard_normal(max(4, n // 2))
    ref = np.convolve(a, b)
    got = cf._conv1d(a.copy(), b.copy())
    check("conv1d_is_direct_convolution__n%d" % n,
          got.shape == ref.shape and np.array_equal(got, ref),
          "_conv1d differs from np.convolve at n=%d: max|delta| = %.3g"
          % (n, np.max(np.abs(got - ref)) if got.shape == ref.shape
             else float('nan')))

# =========================================================== 4. shared buffers
mm._EYE_CACHE.clear()
mm._ZERO_CACHE.clear()
mm.MMSE_FOM(**kw, idx=None)
eye_before = {k: v.copy() for k, v in mm._EYE_CACHE.items()}
zero_before = {k: v.copy() for k, v in mm._ZERO_CACHE.items()}
for _ in range(3):
    mm.MMSE_FOM(**kw, idx=None)
check("shared_eye_zero_buffers_are_not_written_through",
      all(np.array_equal(mm._EYE_CACHE[k], v) for k, v in eye_before.items())
      and all(np.array_equal(mm._ZERO_CACHE[k], v)
              for k, v in zero_before.items()),
      "MMSE_FOM wrote into its cached np.eye/np.zeros buffers; they are shared "
      "across every call, so the next solve starts from corrupted data")

check("repeated_calls_are_deterministic",
      np.isclose(mm.MMSE_FOM(**kw, idx=None)[1], r_self[1],
                 rtol=0, atol=0),
      "MMSE_FOM is not repeatable across calls with identical inputs -- some "
      "state is leaking between invocations")

finish()
