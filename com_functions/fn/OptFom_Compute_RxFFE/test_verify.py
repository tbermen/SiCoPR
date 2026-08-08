"""Verification tests for OptFom_Compute_RxFFE().

# ============================================================
# MATLAB GROUND TRUTH (lines 3305-3369)
# MMSE path (OP.FFE_OPT_METHOD=='MMSE'): get_PSDs -> MMSE; THIS.C/FOM/MMSE_results
#   come from MMSE_results. force path (else): force() -> C; FOM=0, MMSE_results=None.
# Legality check (only when not RxFFE_with_MMSE): RXFFE_Illegal(C) True -> skip_it=1.
# On success the RxFFE taps are applied to the pulse via FFE().
# get_PSDs / MMSE / force / RXFFE_Illegal / FFE / S_RN / S_IN / H_interp are
# top-level fns in the assembled com.py; here they are injected as spies so the
# dispatch/branching logic is unit-tested in isolation (each helper has its own
# test directory, and the whole chain is exercised by the end-to-end run).
# ============================================================
"""
import pytest
import numpy as np
import sys, os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com_functions.fn.OptFom_Compute_RxFFE.py_impl as _mod
from com_functions.fn.OptFom_Compute_RxFFE.py_impl import OptFom_Compute_RxFFE


def _THIS():
    return SimpleNamespace(
        PSD_results=None, txffe=np.array([0.0, 1.0, 0.0]),
        cursor_i=40, g_dc=0.0, g_DC_low=0.0,
        C=None, floating_tap_locations=None, FOM=0.0, MMSE_results=None,
    )


def _param():
    return SimpleNamespace(samples_per_ui=8, RxFFE_cmx=2, RxFFE_cpx=2)


def _chdata():
    return [SimpleNamespace(faxis=np.linspace(0, 50e9, 100))]


def _inject_common(monkeypatch, illegal=False):
    """Inject the bare-global helpers as spies. Returns the FFE call counter."""
    ffe_calls = {'n': 0}

    def spy_FFE(C, cmx, M, sbr):
        ffe_calls['n'] += 1
        return np.asarray(sbr)

    monkeypatch.setattr(_mod, 'FFE', spy_FFE, raising=False)
    monkeypatch.setattr(_mod, 'RXFFE_Illegal', lambda C, param: illegal, raising=False)
    # referenced when building the get_PSDs kwargs (MMSE path)
    for name in ('S_RN', 'S_IN', 'H_interp'):
        monkeypatch.setattr(_mod, name, object(), raising=False)
    return ffe_calls


def test_mmse_path_uses_mmse_results(monkeypatch):
    """MMSE method: THIS.C / FOM / MMSE_results come from the MMSE solver."""
    ffe_calls = _inject_common(monkeypatch)
    monkeypatch.setattr(_mod, 'get_PSDs', lambda *a, **k: SimpleNamespace(), raising=False)
    mmse_C = np.array([0.0, 1.0, 0.0])
    monkeypatch.setattr(_mod, 'MMSE', lambda *a, **k: SimpleNamespace(
        C=mmse_C, floating_tap_locations=np.array([2]), FOM=5.0), raising=False)

    THIS = _THIS()
    OP = SimpleNamespace(FFE_OPT_METHOD='MMSE', RxFFE_with_MMSE=True)
    sbr, THIS, skip_it = OptFom_Compute_RxFFE(np.zeros(100), THIS, None, _chdata(), _param(), OP)

    assert skip_it == 0
    np.testing.assert_array_equal(THIS.C, mmse_C)
    assert THIS.FOM == 5.0
    assert THIS.MMSE_results is not None
    assert ffe_calls['n'] == 1  # taps applied to the pulse


def test_force_path_sets_zero_fom(monkeypatch):
    """Non-MMSE method: C comes from force(); FOM=0, MMSE_results=None."""
    ffe_calls = _inject_common(monkeypatch)
    force_C = np.array([0.0, 1.0, 0.0])
    monkeypatch.setattr(_mod, 'force',
                        lambda *a, **k: (None, force_C, np.array([2])), raising=False)

    THIS = _THIS()
    OP = SimpleNamespace(FFE_OPT_METHOD='ZF', RxFFE_with_MMSE=False)
    sbr, THIS, skip_it = OptFom_Compute_RxFFE(np.zeros(100), THIS, None, _chdata(), _param(), OP)

    assert skip_it == 0
    np.testing.assert_array_equal(THIS.C, force_C)
    assert THIS.FOM == 0
    assert THIS.MMSE_results is None
    assert ffe_calls['n'] == 1


def test_illegal_rxffe_skips(monkeypatch):
    """RXFFE_Illegal True (and not RxFFE_with_MMSE) -> skip_it=1, FFE not applied."""
    ffe_calls = _inject_common(monkeypatch, illegal=True)
    monkeypatch.setattr(_mod, 'force',
                        lambda *a, **k: (None, np.array([9.0, 9.0]), None), raising=False)

    THIS = _THIS()
    OP = SimpleNamespace(FFE_OPT_METHOD='ZF', RxFFE_with_MMSE=False)
    sbr, THIS, skip_it = OptFom_Compute_RxFFE(np.zeros(100), THIS, None, _chdata(), _param(), OP)

    assert skip_it == 1
    assert ffe_calls['n'] == 0  # illegal -> returns before applying FFE


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
