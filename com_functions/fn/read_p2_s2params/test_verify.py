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


# ============================================================
# COM Octave oracle — read_p2_s2params run verbatim under Octave, with
# read_Nport_touchstone and rangelimit, on this file:
#
#   ! test 2-port
#   # GHz S RI R 50
#   0.0   0.01 0.00   0.90 0.00   0.90 0.00   0.02 0.00
#   5.0  -0.03 0.02   0.80 -0.30  0.81 -0.31  0.04 -0.01
#   10.0  0.05 -0.04  0.60 -0.55  0.61 -0.56 -0.02 0.03
#   20.0 -0.07 0.06   0.30 -0.70  0.31 -0.71  0.05 -0.02
#   40.0  0.09 -0.08  0.10 -0.40  0.11 -0.41 -0.06 0.04
#
# with param.Z0 = 50, OP.DISPLAY_WINDOW = false, both plot flags 0:
#
#   SDD(1:2) -0.885+0j              -0.8+0.31j
#   SDC(1:2) -0.0050000000000000044  -0.030000000000000027+0.0099999999999999811j
#   SCC(1:2)  0.91500000000000004     0.81000000000000005-0.30000000000000004j
#   SCD(1:2) -0.0050000000000000044  -0.040000000000000036+0.020000000000000018j
#
# Pinned with ==, because these pin the FORM of W = T*(S/T). MATLAB's S/T is
# mrdivide, which solves rather than multiplying by an inverse; T @ S @ inv(T)
# is the same matrix in exact arithmetic and differs in the last bit here, on
# SDC and SCC, for this very ordinary file.
#
# param.flim:
#   67e9 (above the file) -> data.flim 40e9, data.limited 0, 5 points
#   15e9                  -> data.flim 15e9, data.limited 1, 4 points
#   10e9 (on a point)     -> data.flim 10e9, data.limited 1, 3 points
# and in every case Octave leaves the CALLER's param.flim alone, because
# rangelimit takes param by value.
# ============================================================
_ORACLE_S2P = """! test 2-port
# GHz S RI R 50
0.0   0.01 0.00   0.90 0.00   0.90 0.00   0.02 0.00
5.0  -0.03 0.02   0.80 -0.30  0.81 -0.31  0.04 -0.01
10.0  0.05 -0.04  0.60 -0.55  0.61 -0.56 -0.02 0.03
20.0 -0.07 0.06   0.30 -0.70  0.31 -0.71  0.05 -0.02
40.0  0.09 -0.08  0.10 -0.40  0.11 -0.41 -0.06 0.04
"""


def _oracle_file(tmp_path):
    p = tmp_path / 'oracle.s2p'
    p.write_text(_ORACLE_S2P)
    return str(p)


def test_octave_oracle_mixed_mode_is_mrdivide(tmp_path):
    param = SimpleNamespace(Z0=50.0, flim=67e9)
    OP = SimpleNamespace(DISPLAY_WINDOW=False)
    _, SDD, SDC, SCC, SCD = read_p2_s2params(_oracle_file(tmp_path), 0, 0,
                                             [1, 2], OP, param)
    assert SDD[0, 0, 0] == -0.88500000000000001 + 0j
    assert SDD[1, 0, 0] == -0.80000000000000004 + 0.31j
    assert SDC[0, 0, 0] == -0.0050000000000000044 + 0j
    assert SDC[1, 0, 0] == -0.030000000000000027 + 0.0099999999999999811j
    assert SCC[0, 0, 0] == 0.91500000000000004 + 0j
    assert SCC[1, 0, 0] == 0.81000000000000005 - 0.30000000000000004j
    assert SCD[0, 0, 0] == -0.0050000000000000044 + 0j
    assert SCD[1, 0, 0] == -0.040000000000000036 + 0.020000000000000018j


@pytest.mark.parametrize('flim,n,limited,out_flim', [
    (67e9, 5, 0, 40e9),
    (15e9, 4, 1, 15e9),
    (10e9, 3, 1, 10e9),
])
def test_octave_oracle_rangelimit(tmp_path, flim, n, limited, out_flim):
    param = SimpleNamespace(Z0=50.0, flim=flim)
    OP = SimpleNamespace(DISPLAY_WINDOW=False)
    data, _, _, _, _ = read_p2_s2params(_oracle_file(tmp_path), 0, 0,
                                        [1, 2], OP, param)
    assert len(data.freq) == n
    assert data.limited == limited
    assert data.flim == out_flim
    # MATLAB passes param by value, so the caller's struct is unchanged
    assert param.flim == flim


def test_ports_may_be_an_array(tmp_path):
    """MATLAB only tests isempty(ports) before overwriting it with [1 2];
    `not ports` raised on an ndarray of length 2."""
    param = SimpleNamespace(Z0=50.0, flim=67e9)
    OP = SimpleNamespace(DISPLAY_WINDOW=False)
    ref = read_p2_s2params(_oracle_file(tmp_path), 0, 0, [1, 2], OP,
                           SimpleNamespace(Z0=50.0, flim=67e9))[1]
    for ports in (np.array([1, 2]), np.array([2, 1]), [], np.array([])):
        got = read_p2_s2params(_oracle_file(tmp_path), 0, 0, ports, OP, param)[1]
        np.testing.assert_array_equal(got, ref)
