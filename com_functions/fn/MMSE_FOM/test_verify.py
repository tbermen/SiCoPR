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
