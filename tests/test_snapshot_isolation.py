"""A recorded best must not change when later candidates are evaluated.

Defect #9 in docs/FIX_SUMMARY.md: `BEST.PSD_results = THIS.PSD_results` copies a
struct BY VALUE in MATLAB and binds a REFERENCE in Python. `get_PSDs` mutates its
result argument in place and `optimize_fom` reuses one object per CTLE block, so
the reported noise came from the LAST tick swept rather than the winner's. It
cost 0.18 dB and two pass/fail disagreements, and no unit test could see it --
each tests one function in isolation, while the damage happens to a value
recorded earlier and corrupted later.

The invariant this file checks needs no MATLAB results, only MATLAB's
*semantics*, which are a property of the language: **assignment copies, so a
snapshot taken at candidate N cannot be altered by candidate N+1.** Any Python
alias is a fidelity divergence; it is merely *harmless* when nothing later
mutates the object, and that is not a property anyone should have to verify by
hand on every future edit.

Method, which is why this generalises rather than re-testing one known bug:

  1. call the snapshot function to record a "best"
  2. deep-copy what it recorded
  3. mutate every mutable field of the live candidate IN PLACE, as the next
     candidate's evaluation would
  4. assert the recorded best is unchanged

Step 3 discovers which fields alias instead of requiring a maintained list, so a
new `BEST.x = THIS.x` added by a future feature is caught the day it lands.

`tests/test_reference_leaks.py` covers a DIFFERENT shape of the same defect
class -- writing to a parameter the function never returns (defect #8). Neither
subsumes the other.

Run: python tests/test_snapshot_isolation.py
"""
import copy
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish  # noqa: E402
import sicopr  # noqa: E402


def _mutate_in_place(obj, seen=None):
    """Corrupt every mutable value reachable from obj, in place.

    Stands in for what evaluating the next candidate does to the live struct.
    Returns the number of objects actually mutated, so a vacuous pass (nothing
    was mutable, so of course nothing changed) is detectable.
    """
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    n = 0
    if isinstance(obj, np.ndarray):
        if obj.size and obj.dtype.kind in 'fciub':
            obj.flat[0] = obj.flat[0] + 7919 if obj.dtype.kind != 'b' else (not obj.flat[0])
            n += 1
        return n
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            n += _mutate_in_place(v, seen)
        if obj and isinstance(obj[0], (int, float)):
            obj[0] = obj[0] + 7919
            n += 1
        return n
    if isinstance(obj, dict):
        for v in obj.values():
            n += _mutate_in_place(v, seen)
        return n
    if isinstance(obj, SimpleNamespace):
        for v in vars(obj).values():
            n += _mutate_in_place(v, seen)
        return n
    return n


def _freeze(obj, depth=0):
    """A comparable, hashable-ish rendering of a value for equality checking."""
    if depth > 6:
        return '<deep>'
    if isinstance(obj, np.ndarray):
        return ('ndarray', obj.shape, obj.tobytes() if obj.size < 100000 else obj.sum())
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v, depth + 1) for v in obj)
    if isinstance(obj, dict):
        return tuple((k, _freeze(v, depth + 1)) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])))
    if isinstance(obj, SimpleNamespace):
        return tuple((k, _freeze(v, depth + 1)) for k, v in sorted(vars(obj).items()))
    return obj


def _fixture():
    """A THIS/param/OP/chdata set exercising every branch of the snapshot."""
    n = 160          # long enough that cursor_i = 32 is a valid position
    arr = lambda seed: np.arange(n, dtype=float) + seed        # noqa: E731

    psd = SimpleNamespace(S_tn=arr(1), S_jn=arr(2), S_rj_jn=arr(3), S_xn=arr(4),
                          S_tn_rms=0.5, S_rj_rms=0.25)
    mmse = SimpleNamespace(C=arr(5), sigma_e=0.1, floating_tap_locations=np.array([2, 3]),
                           blim=arr(6), Nw=4)

    THIS = SimpleNamespace(
        txffe=arr(10), tx_index_vector=np.array([1, 2, 3]), ctle_index=2,
        g_dc=-8.0, g_LP_index=1, FOM=12.5, cursor_i=32, itick=-3,
        sigma_N=arr(11), h_J=arr(12), A_s=0.4, A_p=0.5, ISI_N=arr(13),
        tail_RSS=0.01, dfetaps=arr(14),
        floating_tap_locations=np.array([5, 6, 7]), floating_tap_coef=arr(15),
        C=arr(16), PSD_results=psd, MMSE_results=mmse,
        H_ctf=arr(17),
    )
    param = SimpleNamespace(
        current_ffegain=1.0, samples_per_ui=32, cursor_index=1,
        use_bmax=arr(20), use_bmin=arr(21), bmax=arr(22), bmin=arr(23),
        Floating_DFE=True, Floating_RXFFE=True, ndfe=4, ndfe_passed=4,
        RxFFE_cmx=2, RxFFE_cpx=2, bmaxg=0.2, dfe_delta=0.0,
    )
    OP = SimpleNamespace(TDMODE=True, RxFFE=True)
    chdata = [SimpleNamespace(ctle_imp_response=arr(30), sdd21ctf=arr(31))]
    sbr = arr(40)
    return THIS, param, OP, chdata, sbr


def _run(fn, label):
    THIS, param, OP, chdata, sbr = _fixture()
    BEST = SimpleNamespace(FOM=float('-inf'), cursor_i=None)
    try:
        BEST = fn(BEST, THIS, sbr, chdata, param, OP)
    except Exception as e:                       # noqa: BLE001
        check('%s_callable' % label, False,
              'could not exercise the snapshot function: %s: %s' % (type(e).__name__, e))
        return

    recorded = {k: _freeze(v) for k, v in vars(BEST).items()}

    # what evaluating the next candidate does to the live struct
    n_mutated = _mutate_in_place(THIS)
    n_mutated += _mutate_in_place(param)
    check('%s_fixture_is_not_vacuous' % label, n_mutated > 0,
          'nothing mutable was mutated, so the check below proves nothing')

    after = {k: _freeze(v) for k, v in vars(BEST).items()}
    drifted = sorted(k for k in recorded if recorded[k] != after.get(k))

    check('%s_snapshot_survives_next_candidate' % label, not drifted,
          'BEST fields changed when the live candidate was mutated, so they are '
          'REFERENCES to it, not copies: %s. MATLAB assigns structs by value, so '
          'the recorded best cannot move -- this is the defect that made COM '
          'report the last tick swept instead of the winner (FIX_SUMMARY #9). '
          'Fix by copying on assignment (_value_copy), not by arguing the object '
          'happens never to be mutated today.' % ', '.join(drifted))


def main():
    _run(sicopr.OptFom_Update_Best_Setttings, 'update_best')
    _run(sicopr.OptFom_Update_Best_Settings_EQ_Failed, 'update_best_eq_failed')
    return finish()


if __name__ == '__main__':
    sys.exit(main())
