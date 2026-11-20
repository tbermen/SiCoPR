"""Verification tests for optimize_fom_BestEQ (new in 4p18p0).

COM Octave CALL TRACE: the reference's optimize_fom_BestEQ, extracted verbatim
from octave/com_ieee8023_4p18p0_octave_compat.m, run with optimize_fom replaced
by a stub that records the param it is handed and returns sentinel index fields.
Generator: runs/4p18p0_oracles/gen_besteq_trace.py (local). The whole function,
optimize_fom included, is checked against COM Octave by whole cases that trigger
it (Clip Method = Fast; Defer_Floating_RxFFE), docs/MATLAB_4p18p0_CHANGES.md.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from com_functions.fn.optimize_fom_BestEQ.py_impl import optimize_fom_BestEQ  # noqa: E402

# COM Octave, the trace: what the stub optimize_fom received, and what came back
_OCT_RECEIVED = {
    'cursor_gain': [0.75], 'ctle_gdc_values': [-2.0], 'g_DC_HP_values': [-2.0],
    'ts_sample_adj_range': [-3.0, -3.0], 'tx_ffe_cm1_values': [0.1],
    'tx_ffe_cm2_values': [-0.04], 'tx_ffe_cp1_values': [-0.08], 'untouched': [42.0],
}
_OCT_RETURNED = {'ctle': 2, 'best_G_high_pass': 3, 'cur': 3,
                 'txffe': [-0.04, 0.1, 0.78, -0.08], 'itick': 77, 'FOM': 12.5}


def _inputs():
    fr = SimpleNamespace(best_current_ffegain=0.75, ctle=2, best_G_high_pass=3, itick=-3,
                         txffe=np.array([-0.04, 0.1, 0.78, -0.08]), cur=3)
    param = SimpleNamespace(cursor_gain=np.array([0.5, 0.75, 1]),
                            ctle_gdc_values=np.array([-1., -2., -3.]),
                            g_DC_HP_values=np.array([0., -1., -2.]),
                            ts_sample_adj_range=np.array([-8., 8.]),
                            tx_ffe_cm1_values=np.array([0, 0.05, 0.1]),
                            tx_ffe_cm2_values=np.array([0, 0.02]),
                            tx_ffe_cp1_values=np.array([0, -0.05, -0.1]), untouched=42)
    return fr, param


def _run():
    seen = {}

    def stub(OP, param, chdata, sigma_bn, do_C2M):
        seen['param'] = param
        seen['args'] = (OP, chdata, sigma_bn, do_C2M)
        return SimpleNamespace(ctle=1, best_G_high_pass=1, cur=99, txffe=9, itick=77, FOM=12.5)

    fr, param = _inputs()
    OP, chdata = SimpleNamespace(x=1), [0]
    out = optimize_fom_BestEQ(fr, OP, param, chdata, 0.003, 1, _optimize_fom_fn=stub)
    return out, seen, param, OP, chdata


def test_narrowed_sweep_matches_com_octave():
    _, seen, _, OP, chdata = _run()
    for k, want in _OCT_RECEIVED.items():
        got = np.atleast_1d(np.asarray(getattr(seen['param'], k), dtype=float)).ravel().tolist()
        assert got == want, (k, got, want)
    assert seen['args'] == (OP, chdata, 0.003, 1)


def test_narrowed_grids_are_one_element_vectors():
    """MATLAB's x(k) is 1x1, and OptFom_Calculate_Settings takes length() of
    both grids. A bare numpy scalar crashed the 4p18p0 Clip Method = Fast case
    there (len() of unsized object), 2026-11-19."""
    _, seen, _, _, _ = _run()
    for k in ('ctle_gdc_values', 'g_DC_HP_values'):
        v = getattr(seen['param'], k)
        assert np.ndim(v) == 1 and len(v) == 1, (k, v)


def test_index_fields_restored_as_com_octave():
    out, _, _, _, _ = _run()
    for k, want in _OCT_RETURNED.items():
        got = np.asarray(getattr(out, k)).ravel().tolist()
        assert got == np.ravel(want).tolist(), (k, got, want)


def test_callers_param_is_not_narrowed():
    """MATLAB passes param by value; the caller's grids must survive."""
    _, seen, param, _, _ = _run()
    assert seen['param'] is not param
    assert param.ctle_gdc_values.tolist() == [-1., -2., -3.]
    assert param.ts_sample_adj_range.tolist() == [-8., 8.]
    assert param.tx_ffe_cm1_values.tolist() == [0, 0.05, 0.1]
