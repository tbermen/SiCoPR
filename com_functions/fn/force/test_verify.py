"""Tests for force (RxFFE tap solver).

MATLAB GROUND TRUTH:
  force(V, param, OP, ix, C=[]) solves for FFE taps.
  With pre-set C: Vfiltered = FFE(C, cmx, spui, V).
  With C=[]: solves VV*C = FV where FV[cmx]=V[ix].
  Tap constraint 'unity cursor': Cmod[cmx] = 1.
  FFE with 1 tap [1.0] at cmx=0: Vfiltered == V.
  FFE with cursor-only: Vfiltered ≈ V (identity filter).
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.force.py_impl import force, _FFE


def _make_param(cmx=0, cpx=0, spui=32, ndfe=0, bmax=None):
    p = SimpleNamespace()
    p.RxFFE_cmx = cmx
    p.RxFFE_cpx = cpx
    p.samples_per_ui = spui
    p.ndfe = ndfe
    p.bmax = bmax if bmax is not None else np.array([1.0])
    p.N_bg = 0
    p.RxFFE_stepz = 0
    return p


def _make_OP(constraint='unity cursor', method='FORCE'):
    op = SimpleNamespace()
    op.RXFFE_TAP_CONSTRAINT = constraint
    op.FFE_OPT_METHOD = method
    return op


def test_ffe_identity_single_tap():
    """FFE with single tap C=[1] and cmx=0: output equals input."""
    V = np.sin(np.linspace(0, 2 * np.pi, 64))
    out = _FFE(np.array([1.0]), 0, 1, V)
    assert np.allclose(out, V)


def test_ffe_passthrough_with_preset_C():
    """force with preset C=[1.0], cmx=0, cpx=0: Vfiltered = V."""
    V = np.random.randn(128)
    param = _make_param(cmx=0, cpx=0, spui=1)
    Vfiltered, Cmod, _ = force(V, param, _make_OP(), C=np.array([1.0]))
    assert np.allclose(Vfiltered, V)


def test_returns_three_outputs():
    """force always returns (Vfiltered, Cmod, idx)."""
    V = np.ones(64)
    param = _make_param(cmx=1, cpx=1, spui=4)
    result = force(V, param, _make_OP())
    assert len(result) == 3


def test_unity_cursor_constraint():
    """Tap constraint 'unity cursor': Cmod[cmx] should be 1."""
    np.random.seed(42)
    V = np.random.randn(128)
    param = _make_param(cmx=2, cpx=1, spui=8)
    op = _make_OP(constraint='unity cursor')
    _, Cmod, _ = force(V, param, op)
    assert abs(float(Cmod[2]) - 1.0) < 1e-6


def test_cmod_length_matches_num_taps():
    """len(Cmod) = cmx + cpx + 1."""
    V = np.random.randn(256)
    param = _make_param(cmx=2, cpx=3, spui=16)
    _, Cmod, _ = force(V, param, _make_OP())
    assert len(Cmod) == 6   # 2+3+1


def test_return_v_false_skips_filter():
    """return_V=0: Vfiltered is empty array."""
    V = np.random.randn(64)
    param = _make_param(cmx=0, cpx=0, spui=4)
    Vfiltered, _, _ = force(V, param, _make_OP(), return_V=0)
    assert len(Vfiltered) == 0


def test_wiener_hopf_raises():
    """FFE_OPT_METHOD='WIENER-HOPF' raises NotImplementedError."""
    V = np.random.randn(64)
    param = _make_param(cmx=1, cpx=1, spui=4)
    op = _make_OP(method='WIENER-HOPF')
    with pytest.raises(NotImplementedError):
        force(V, param, op)
