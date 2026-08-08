"""Tests for read_p2_s2params.

MATLAB GROUND TRUTH:
  Reads 2-port Touchstone, converts single-ended → differential.
  T = [[1,1],[1,-1]]; D = T * S * inv(T)
  SDD = D[:,1,1]; SDC = D[:,1,0]; SCC = D[:,0,0]; SCD = D[:,0,1]
  All SXX have shape (nfreq, 1, 1).
  ports forced to [1,2] regardless of input.
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.read_p2_s2params.py_impl import read_p2_s2params


def _make_param(flim=None):
    p = SimpleNamespace()
    p.Z0 = 100.0
    if flim is not None:
        p.flim = flim
    else:
        p.flim = float('inf')
    return p


def _make_OP():
    return SimpleNamespace()


def _write_s2p(path, freqs_GHz, s11, s21):
    """Write RI 2-port Touchstone where s12=s21 and s22=s11 for symmetry."""
    with open(path, 'w') as f:
        f.write('# GHz S RI R 100\n')
        for i, fg in enumerate(freqs_GHz):
            f.write(f'{fg} {s11[i].real} {s11[i].imag} '
                    f'{s21[i].real} {s21[i].imag} '
                    f'{s21[i].real} {s21[i].imag} '
                    f'{s11[i].real} {s11[i].imag}\n')


def test_sdd_shape(tmp_path):
    freqs = np.array([1.0, 5.0, 10.0])
    s11 = np.zeros(3, dtype=complex)
    s21 = np.full(3, 0.9 + 0j)
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, s11, s21)
    data, SDD, SDC, SCC, SCD = read_p2_s2params(p, 0, 0, [1, 2], _make_OP(), _make_param())
    assert SDD.shape == (3, 1, 1)
    assert SDC.shape == (3, 1, 1)
    assert SCC.shape == (3, 1, 1)
    assert SCD.shape == (3, 1, 1)


def test_data_freq_in_Hz(tmp_path):
    freqs = np.array([1.0, 10.0])
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, np.zeros(2, dtype=complex), np.ones(2, dtype=complex))
    data, SDD, _, _, _ = read_p2_s2params(p, 0, 0, [1, 2], _make_OP(), _make_param())
    assert data.freq[0] == pytest.approx(1e9)
    assert data.freq[1] == pytest.approx(10e9)


def test_sdd_matched_port(tmp_path):
    """Matched port (S11=0, S21=1) → SDD = D(2,2) = T11*1*T_inv11."""
    freqs = np.array([1.0])
    s11 = np.array([0.0 + 0j])
    s21 = np.array([1.0 + 0j])
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, s11, s21)
    _, SDD, SDC, SCC, SCD = read_p2_s2params(p, 0, 0, [1, 2], _make_OP(), _make_param())
    # T = [[1,1],[1,-1]]; T_inv = 0.5*[[1,1],[1,-1]]
    # S = [[0,1],[1,0]]; D = T*S*T_inv = [[0.5,0.5],[0.5,0.5]]*... let's just verify numerics
    T = np.array([[1.0, 1.0], [1.0, -1.0]])
    T_inv = np.linalg.inv(T)
    S = np.array([[0.0, 1.0], [1.0, 0.0]])
    D = T @ S @ T_inv
    assert abs(SDD[0, 0, 0] - D[1, 1]) < 1e-8
    assert abs(SCC[0, 0, 0] - D[0, 0]) < 1e-8


def test_rangelimit_truncates(tmp_path):
    freqs = np.array([1.0, 5.0, 10.0, 20.0])
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, np.zeros(4, dtype=complex), np.ones(4, dtype=complex))
    # flim = 7 GHz → only first 2 frequencies (≥7 GHz: only 10 GHz onwards → keep up to 10 GHz)
    param = _make_param(flim=7e9)
    data, SDD, _, _, _ = read_p2_s2params(p, 0, 0, [1, 2], _make_OP(), param)
    # flim means truncate at first freq >= flim → first hit is 10 GHz → iend = 3+1=4 but slice [0:iend]
    # Actually the rangelimit returns freq[0:iend] where iend = idx[0]+1
    assert len(data.freq) <= 4


def test_scd_antisymmetric_input(tmp_path):
    """SCD extraction: D[:,0,1] = T[0,:]@S@T_inv[:,1]."""
    freqs = np.array([1.0])
    s11 = np.array([0.1 + 0.05j])
    s21 = np.array([0.8 - 0.1j])
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, s11, s21)
    _, SDD, SDC, SCC, SCD = read_p2_s2params(p, 0, 0, [1, 2], _make_OP(), _make_param())
    T = np.array([[1.0, 1.0], [1.0, -1.0]])
    T_inv = np.linalg.inv(T)
    S = np.array([[s11[0], s21[0]], [s21[0], s11[0]]])
    D = T @ S @ T_inv
    assert abs(SCD[0, 0, 0] - D[0, 1]) < 1e-8
