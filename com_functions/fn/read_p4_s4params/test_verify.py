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


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py; read_p4_s4params,
# rangelimit and auto_port_order are byte-identical between
# octave/com_ieee8023_4p16p0_octave_compat.m and matlab/com_ieee8023_4p16p0.m.
# read_Nport_touchstone is the compat file's repaired reader -- the MATLAB one
# uses textscan, which Octave silently truncates -- and that reader is
# documented to return the same S-parameters bit for bit.)
#
# Fixture: 4 frequencies (0, 10, 25, 50 GHz), S(i,j) deliberately NOT
# symmetric so the D-matrix index mapping is exercised, RI format, R 50,
# ports [1 3 2 4], Z0 50, flim 100 GHz (past the end of the file),
# Txpskew=3 ps, Txnskew=-1 ps, Rxpskew=2.5 ps, Rxnskew=0.5 ps.
#
# What these catch:
#   1. MATLAB's Sigfct is `@(sigma2,sigma1,sigma4,sigma3)...` -- the parameter
#      names are transposed on purpose -- so the skews bind
#      sigma1=Txn, sigma2=Txp, sigma3=Rxn, sigma4=Rxp. The port took them in
#      call order, which silently swapped 1<->2 and 3<->4 in the sigma matrix
#      whenever a p skew differed from its n skew (|dSDC| up to 0.1255).
#   2. `Snew / T` is mrdivide, a solve, not a multiply by inv(T).
#   3. rangelimit's param is passed BY VALUE; the private copy wrote
#      param.flim back into the caller's object.
# ============================================================

def _oracle_s(k):
    S = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            S[i, j] = ((0.1 * (i + 1) + 0.01 * (j + 1)) / (1 + 0.3 * k)
                       + 1j * (0.02 * (i + 1) - 0.03 * (j + 1)) * (1 + 0.1 * k))
    return S


def _write_oracle_s4p(path):
    rows = []
    for k, fg in enumerate([0.0, 10.0, 25.0, 50.0]):
        S = _oracle_s(k)
        row = ['%.17g' % fg]
        for i in range(4):
            for j in range(4):
                row += ['%.17g' % S[i, j].real, '%.17g' % S[i, j].imag]
        rows.append(' '.join(row))
    with open(path, 'w') as f:
        f.write('# GHz S RI R 50\n' + '\n'.join(rows) + '\n')


def _oracle_param():
    return SimpleNamespace(Z0=50.0, flim=100e9, Txpskew=3.0, Txnskew=-1.0,
                           Rxpskew=2.5, Rxnskew=0.5)


def _run_oracle_case(tmp_path):
    p = str(tmp_path / 'fixture.s4p')
    _write_oracle_s4p(p)
    param = _oracle_param()
    out = read_p4_s4params(p, 0, 0, [1, 3, 2, 4],
                           SimpleNamespace(DISPLAY_WINDOW=0), param)
    return out + (param,)


def test_octave_skewed_mixed_mode(tmp_path):
    """COM Octave, the asymmetric-skew fixture above, third frequency."""
    data, SDD, SDC, SCC, SCD, ports, _ = _run_oracle_case(tmp_path)
    assert list(ports) == [1, 3, 2, 4]
    np.testing.assert_allclose(data.freq, [0.0, 10e9, 25e9, 50e9])
    np.testing.assert_allclose(SDD[2].ravel(), [
        -0.032170553032775787 + 0.036856464388770618j,
        -0.0099320706699547873 + 0.026423876089297249j,
        -0.036814532001131786 + 0.028058824918477748j,
        -0.016135221446448689 + 0.017599923348547073j], rtol=1e-12)
    np.testing.assert_allclose(SDC[2].ravel(), [
        -0.079295103639119721 - 0.16362413819436167j,
        -0.089593198765815674 - 0.18466005323126555j,
        -0.066561629068211337 - 0.14455614293989621j,
        -0.067296645166451813 - 0.16035136734195049j], rtol=1e-12)
    np.testing.assert_allclose(SCC[2].ravel(), [
        0.24420280467838895 + 0.076185425059713777j,
        0.27369177389836907 + 0.034665667922157584j,
        0.35226547577827622 + 0.17017215344708619j,
        0.38409228576250043 + 0.14071853474954765j], rtol=1e-12)
    np.testing.assert_allclose(SCD[2].ravel(), [
        -0.0093832848809085774 - 0.014732944371761664j,
        -0.03216097759250254 + 0.016097580822253696j,
        0.014684503693481302 - 0.050344332044631956j,
        -0.021537271164006 - 0.0023566532186473331j], rtol=1e-12)


def test_octave_rangelimit_does_not_write_back_to_the_caller(tmp_path):
    """MATLAB passes param by value, so rangelimit's param.flim assignment
    stays inside rangelimit's own copy.

    COM Octave, a 50 GHz file read with param.flim = 100e9: data.flim comes
    back 50e9 and data.limited 0, while the caller's param.flim is still
    100e9. The private copy overwrote the caller's struct.
    """
    data, _, _, _, _, _, param = _run_oracle_case(tmp_path)
    assert data.flim == pytest.approx(50e9)
    assert data.limited == 0
    assert param.flim == pytest.approx(100e9)
