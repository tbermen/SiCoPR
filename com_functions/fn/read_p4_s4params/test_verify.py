"""Tests for read_p4_s4params.

MATLAB GROUND TRUTH:
  Reads 4-port Touchstone, applies TX/RX skew, converts to mixed-mode.
  T = 4×4 mixed-mode transform; D = T * sigma * S * inv(T).
  SDD/SDC/SCC/SCD shape: (nfreq, 2, 2).
  No skew (Txpskew=Txnskew=Rxpskew=Rxnskew=0): sigma=ones → D = T*S*inv(T).
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.read_p4_s4params.py_impl import read_p4_s4params


def _make_param(flim=None):
    p = SimpleNamespace()
    p.Z0 = 100.0
    p.flim = flim if flim is not None else float('inf')
    p.Txpskew = 0.0
    p.Txnskew = 0.0
    p.Rxpskew = 0.0
    p.Rxnskew = 0.0
    return p


def _make_OP():
    return SimpleNamespace()


def _write_s4p(path, freqs_GHz, S_per_freq, Z0=100.0):
    """Write 4-port Touchstone RI file (row-major per Touchstone 1.x spec).
    S_per_freq: list of (4,4) complex arrays, one per frequency.
    """
    with open(path, 'w') as f:
        f.write(f'# GHz S RI R {Z0}\n')
        for i, fg in enumerate(freqs_GHz):
            S = S_per_freq[i]
            row = f'{fg}'
            for r in range(4):   # row-major: row outer
                for c in range(4):  # column inner
                    v = S[r, c]
                    row += f' {v.real} {v.imag}'
            f.write(row + '\n')


def test_sdd_shape(tmp_path):
    freqs = np.array([1.0, 5.0])
    S = [np.eye(4, dtype=complex) * 0.5 for _ in range(2)]
    p = str(tmp_path / 'test.s4p')
    _write_s4p(p, freqs, S)
    data, SDD, SDC, SCC, SCD, ports = read_p4_s4params(p, 0, 0, [1, 3, 2, 4], _make_OP(), _make_param())
    assert list(ports) == [1, 3, 2, 4]  # r4p15p0: 6th return echoes supplied order
    assert SDD.shape == (2, 2, 2)
    assert SDC.shape == (2, 2, 2)
    assert SCC.shape == (2, 2, 2)
    assert SCD.shape == (2, 2, 2)


def test_frequency_in_Hz(tmp_path):
    freqs = np.array([1.0, 10.0])
    S = [np.zeros((4, 4), dtype=complex) for _ in range(2)]
    p = str(tmp_path / 'test.s4p')
    _write_s4p(p, freqs, S)
    data, _, _, _, _, _ = read_p4_s4params(p, 0, 0, [1, 3, 2, 4], _make_OP(), _make_param())
    assert data.freq[0] == pytest.approx(1e9)
    assert data.freq[1] == pytest.approx(10e9)


def test_no_skew_mixed_mode_identity(tmp_path):
    """With no skew, D = T * S * inv(T) at each frequency."""
    freqs = np.array([1.0])
    S0 = np.zeros((4, 4), dtype=complex)
    S0[1, 0] = 0.8   # S21
    S0[0, 1] = 0.8   # S12
    p = str(tmp_path / 'test.s4p')
    _write_s4p(p, freqs, [S0])
    param = _make_param()
    data, SDD, SDC, SCC, SCD, _ = read_p4_s4params(p, 0, 0, [1, 2, 3, 4], _make_OP(), param)
    T = np.array([[1.0, 1.0, 0.0, 0.0],
                  [1.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0, -1.0]])
    T_inv = np.linalg.inv(T)
    D_expected = T @ S0 @ T_inv
    # SDD[0,0,0] = D(2,2) = D_expected[1,1] (0-based)
    assert abs(SDD[0, 0, 0] - D_expected[1, 1]) < 1e-8


def test_sdd_index_mapping(tmp_path):
    """SDD[:,1,0] = D[:,3,1] (MATLAB D(4,2))."""
    freqs = np.array([1.0])
    S0 = np.zeros((4, 4), dtype=complex)
    S0[3, 1] = 0.5 + 0.1j
    p = str(tmp_path / 'test.s4p')
    _write_s4p(p, freqs, [S0])
    _, SDD, _, _, _, _ = read_p4_s4params(p, 0, 0, [1, 2, 3, 4], _make_OP(), _make_param())
    T = np.array([[1.0, 1.0, 0.0, 0.0],
                  [1.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0, -1.0]])
    T_inv = np.linalg.inv(T)
    D_exp = T @ S0 @ T_inv
    assert abs(SDD[0, 1, 0] - D_exp[3, 1]) < 1e-8


def test_data_has_m_and_freq(tmp_path):
    freqs = np.array([1.0])
    S = [np.zeros((4, 4), dtype=complex)]
    p = str(tmp_path / 'test.s4p')
    _write_s4p(p, freqs, S)
    data, _, _, _, _, _ = read_p4_s4params(p, 0, 0, [1, 3, 2, 4], _make_OP(), _make_param())
    assert hasattr(data, 'm')
    assert hasattr(data, 'freq')
    assert data.m.shape == (1, 4, 4)


def test_empty_ports_triggers_auto_detection(tmp_path):
    """r4p15p0: empty ports -> auto_port_order resolves [1,3,2,4] for 1<->2,3<->4."""
    freqs = np.array([1.0, 5.0, 10.0, 20.0])
    S_per = []
    for _ in freqs:
        S = np.zeros((4, 4), dtype=complex)
        for d in range(4):
            S[d, d] = 0.05
        S[0, 1] = S[1, 0] = 0.8
        S[2, 3] = S[3, 2] = 0.8
        S_per.append(S)
    p = str(tmp_path / 'auto.s4p')
    _write_s4p(p, freqs, S_per)
    _, _, _, _, _, ports = read_p4_s4params(p, 0, 0, [], _make_OP(), _make_param())
    assert list(ports) == [1, 3, 2, 4]
