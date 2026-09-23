"""Verification tests for OptFom_Compute_RxFFE().

# ============================================================
# MATLAB GROUND TRUTH — 4p16p0 lines 3585-3646 (4p15p0 3305-3369)
#
# A dispatcher: 'MMSE' runs get_PSDs then MMSE and takes C / floating taps /
# FOM from MMSE_results; anything else runs force() and reports FOM = 0 with
# no MMSE_results.  RXFFE_Illegal gates the (expensive) FFE, but only when
# OP.RxFFE_with_MMSE is false.
#
# The expected values here were produced by running that reference function
# VERBATIM under Octave (tools/octave_oracle.py) with the five callees
# replaced by the same deterministic stubs used below, so what is compared is
# the dispatcher itself: which branch runs, what each callee is handed, what
# lands in THIS, and what the caller still holds afterwards.  See the
# "COM Octave" block.
#
# The point of driving it rather than reading it: MATLAB passes structs BY
# VALUE.  `OP.WO_TXFFE=0` inside this function is invisible to the caller, and
# the oracle says so — the caller's OP.WO_TXFFE is still 1 on return, in every
# branch.  Python passes OP by reference, so the bare assignment cleared the
# caller's flag for the rest of the run.
# ============================================================
"""
import os
import sys

import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sicopr                                                        # noqa: E402
import com_functions.fn.OptFom_Compute_RxFFE.py_impl as _mod         # noqa: E402
from com_functions.fn.OptFom_Compute_RxFFE.py_impl import OptFom_Compute_RxFFE  # noqa: E402

# S_RN / S_IN / H_interp are only forwarded, never called here; take the real
# ones from the assembled module so nothing stands in for them.
for _n in ('S_RN', 'S_IN', 'H_interp'):
    setattr(_mod, _n, getattr(sicopr, _n))


# ---------------------------------------------------------------------------
# The stubs, mirrored from the .m files handed to Octave
# ---------------------------------------------------------------------------

def _stub_get_PSDs(PSD_results, sbr, cursor_i, txffe, g_dc, g_DC_low,
                   param, chdata, OP, **kw):
    return SimpleNamespace(saw_WO_TXFFE=OP.WO_TXFFE, saw_cursor_i=cursor_i,
                           saw_g_dc=g_dc, saw_g_DC_low=g_DC_low,
                           saw_txffe=np.asarray(txffe), tag=77,
                           S_rn=1, S_tn=2, S_xn=3, S_jn=4, S_qn=5)


def _stub_MMSE(PSD_results, sbr, cursor_i, param, OP):
    return SimpleNamespace(C=np.array([0.25, 1.0, -0.125]),
                           floating_tap_locations=np.array([5, 9]),
                           FOM=12.5, saw_tag=PSD_results.tag,
                           saw_WO_TXFFE=OP.WO_TXFFE, saw_cursor_i=cursor_i,
                           saw_RxFFE_cpx=param.RxFFE_cpx)


def _stub_force(sbr, param, OP, cursor_i, arg5, arg6, chdata, txffe, Noise_XC):
    return -1, np.array([0.5, 1.0, -0.25]), np.array([3, 7])


def _stub_FFE(C, cmx, M, sbr):
    return np.asarray(sbr) * 2 + cmx * 100 + M * 1000 + np.sum(C)


def _install(monkeypatch, illegal):
    monkeypatch.setattr(_mod, 'get_PSDs', _stub_get_PSDs, raising=False)
    monkeypatch.setattr(_mod, 'MMSE', _stub_MMSE, raising=False)
    monkeypatch.setattr(_mod, 'force', _stub_force, raising=False)
    monkeypatch.setattr(_mod, 'FFE', _stub_FFE, raising=False)
    monkeypatch.setattr(_mod, 'RXFFE_Illegal',
                        lambda C, param: bool(param.force_illegal), raising=False)


def _THIS():
    return SimpleNamespace(PSD_results=None, txffe=np.array([0.0, 1.0, 0.0]),
                           cursor_i=40, g_dc=-6.0, g_DC_low=-2.0)


def _param(illegal=False):
    return SimpleNamespace(samples_per_ui=8, RxFFE_cmx=2, RxFFE_cpx=2,
                           force_illegal=int(illegal))


def _OP(method, rxffe_with_mmse):
    return SimpleNamespace(DISPLAY_WINDOW=False, DEBUG=False,
                           FFE_OPT_METHOD=method,
                           RxFFE_with_MMSE=int(rxffe_with_mmse), WO_TXFFE=1)


def _chdata():
    return [SimpleNamespace(faxis=np.array([0.0, 1.0, 2.0]))]


def _run(monkeypatch, method, rxffe_with_mmse, illegal, sbr=None):
    _install(monkeypatch, illegal)
    THIS, param, OP = _THIS(), _param(illegal), _OP(method, rxffe_with_mmse)
    sbr_in = np.array([1.0, 2.0, 3.0]) if sbr is None else sbr
    out = OptFom_Compute_RxFFE(sbr_in.copy(), THIS, None, _chdata(), param, OP)
    return out, OP, THIS


# ===========================================================================
# COM Octave — reference values
#
# The verbatim 4p16p0 OptFom_Compute_RxFFE run under Octave with these
# get_PSDs / MMSE / force / RXFFE_Illegal / FFE stubs on disk:
#
#   OP.FFE_OPT_METHOD='MMSE'; OP.RxFFE_with_MMSE=1; OP.WO_TXFFE=1;
#   param.samples_per_ui=8; param.RxFFE_cmx=2; param.RxFFE_cpx=2;
#   THIS.PSD_results=[]; THIS.txffe=[0 1 0]; THIS.cursor_i=40;
#   THIS.g_dc=-6; THIS.g_DC_low=-2;  sbr=[1 2 3];
#   [sbr_out, THIS_out, skip_it] = OptFom_Compute_RxFFE(sbr, THIS, [], chdata, param, OP)
#
# FFE stub: out = sbr*2 + cmx*100 + M*1000 + sum(C), so sbr_out reads back the
# argument ORDER as well as the taps.
# ===========================================================================

OCT_SBR_MMSE = [8203.125, 8205.125, 8207.125]   # sum(C) = 1.125
OCT_SBR_FORCE = [8203.25, 8205.25, 8207.25]     # sum(C) = 1.25
OCT_C_MMSE = [0.25, 1.0, -0.125]
OCT_C_FORCE = [0.5, 1.0, -0.25]
OCT_FTL_MMSE = [5.0, 9.0]
OCT_FTL_FORCE = [3.0, 7.0]
OCT_FOM_MMSE = 12.5
OCT_FOM_FORCE = 0.0
OCT_CALLER_WO_TXFFE = 1.0        # every branch: the caller's OP is untouched
OCT_GET_PSDS_SAW_WO_TXFFE = 0.0  # ... while the callees do see the 0
OCT_MMSE_SAW_WO_TXFFE = 0.0
# caller's THIS after the call gains no fields; the returned copy carries them
OCT_CALLER_THIS_FIELDS = ('PSD_results', 'txffe', 'cursor_i', 'g_dc', 'g_DC_low')
OCT_THIS_OUT_FIELDS = OCT_CALLER_THIS_FIELDS + (
    'C', 'floating_tap_locations', 'FOM', 'MMSE_results')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['MMSE', 'mmse'])
def test_mmse_path_matches_reference(monkeypatch, method):
    """MMSE branch (also on a lower-case method name: MATLAB upper()s it)."""
    (sbr, THIS, skip_it), OP, _ = _run(monkeypatch, method, True, False)

    assert skip_it == 0
    np.testing.assert_array_equal(sbr, OCT_SBR_MMSE)
    np.testing.assert_array_equal(THIS.C, OCT_C_MMSE)
    np.testing.assert_array_equal(THIS.floating_tap_locations, OCT_FTL_MMSE)
    assert THIS.FOM == OCT_FOM_MMSE
    assert THIS.MMSE_results is not None
    assert THIS.PSD_results.tag == 77
    assert THIS.MMSE_results.saw_tag == 77       # MMSE got get_PSDs' output
    assert set(OCT_THIS_OUT_FIELDS) <= set(vars(THIS))


def test_force_path_matches_reference(monkeypatch):
    """Non-MMSE branch: C from force(), FOM 0, MMSE_results empty, PSD_results
    passed straight through from THIS."""
    (sbr, THIS, skip_it), OP, _ = _run(monkeypatch, 'ZF', False, False)

    assert skip_it == 0
    np.testing.assert_array_equal(sbr, OCT_SBR_FORCE)
    np.testing.assert_array_equal(THIS.C, OCT_C_FORCE)
    np.testing.assert_array_equal(THIS.floating_tap_locations, OCT_FTL_FORCE)
    assert THIS.FOM == OCT_FOM_FORCE
    assert THIS.MMSE_results is None
    assert THIS.PSD_results is None


def test_illegal_rxffe_skips_before_ffe(monkeypatch):
    """RXFFE_Illegal and not RxFFE_with_MMSE: skip_it = 1, sbr comes back
    untouched (FFE never runs) and THIS gains no result fields."""
    calls = {'n': 0}
    _install(monkeypatch, True)

    def counting_FFE(C, cmx, M, sbr):
        calls['n'] += 1
        return _stub_FFE(C, cmx, M, sbr)

    monkeypatch.setattr(_mod, 'FFE', counting_FFE, raising=False)
    THIS, param, OP = _THIS(), _param(True), _OP('ZF', False)
    sbr, THIS, skip_it = OptFom_Compute_RxFFE(
        np.array([1.0, 2.0, 3.0]), THIS, None, _chdata(), param, OP)

    assert skip_it == 1
    np.testing.assert_array_equal(sbr, [1.0, 2.0, 3.0])
    assert calls['n'] == 0
    assert set(vars(THIS)) == set(OCT_CALLER_THIS_FIELDS)


def test_rxffe_with_mmse_bypasses_the_legality_check(monkeypatch):
    """Same illegal taps, but RxFFE_with_MMSE on: the reference does not check,
    so skip_it stays 0 and FFE runs."""
    (sbr, THIS, skip_it), OP, _ = _run(monkeypatch, 'MMSE', True, True)
    assert skip_it == 0
    np.testing.assert_array_equal(sbr, OCT_SBR_MMSE)


def test_wo_txffe_does_not_leak_to_the_caller(monkeypatch):
    """MATLAB passes OP by value: the caller still reads WO_TXFFE = 1.

    optimize_fom sets OP.WO_TXFFE = 1 once, before the TXFFE loop, so that the
    first get_PSDs computes S_rn; every OptFom_Compute_RxFFE inside the loop
    then clears it for its OWN get_PSDs call only.  Clearing the caller's copy
    instead makes that setup silently conditional on loop order.
    """
    (_, THIS, _), OP, _ = _run(monkeypatch, 'MMSE', True, False)
    assert OP.WO_TXFFE == OCT_CALLER_WO_TXFFE
    assert THIS.PSD_results.saw_WO_TXFFE == OCT_GET_PSDS_SAW_WO_TXFFE
    assert THIS.MMSE_results.saw_WO_TXFFE == OCT_MMSE_SAW_WO_TXFFE


def test_wo_txffe_untouched_on_the_force_path(monkeypatch):
    """The reference only assigns WO_TXFFE inside case 'MMSE'."""
    (_, _, _), OP, _ = _run(monkeypatch, 'ZF', False, False)
    assert OP.WO_TXFFE == OCT_CALLER_WO_TXFFE


def test_callee_arguments_match_reference(monkeypatch):
    """get_PSDs takes (PSD_results, sbr, cursor_i, txffe, g_dc, g_DC_low, ...)
    and MMSE takes (PSD_results, sbr, cursor_i, param, OP), both read out of
    THIS.  The FFE argument order is checked through sbr_out."""
    (_, THIS, _), OP, _ = _run(monkeypatch, 'MMSE', True, False)
    p = THIS.PSD_results
    assert p.saw_cursor_i == 40
    assert p.saw_g_dc == -6.0
    assert p.saw_g_DC_low == -2.0
    np.testing.assert_array_equal(p.saw_txffe, [0.0, 1.0, 0.0])
    assert THIS.MMSE_results.saw_cursor_i == 40
    assert THIS.MMSE_results.saw_RxFFE_cpx == 2


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
