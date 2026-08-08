"""Verification tests for read_s4p_files().

# ============================================================
# MATLAB GROUND TRUTH (lines 10594-10767)
# Reads S4P/S2P files; populates chdata with faxis, sddXX_raw,
# sddXX_orig, sdcXX, scdXX, sccXX fields.
# INC_PACKAGE=0: sdd21 = sdd21_raw.copy() (bypass package).
# Validates same frequency axis across all channels.
# S2P: sdd11_raw only; S4P: all four 2×2 mixed-mode sub-matrices.
# ============================================================
"""
import pytest
import numpy as np
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.read_s4p_files.py_impl import read_s4p_files


def _write_s4p(path, freqs_GHz, S_per_freq, Z0=100.0):
    """Write 4-port Touchstone RI file (row-major)."""
    with open(path, 'w') as f:
        f.write(f'# GHz S RI R {Z0}\n')
        for i, fg in enumerate(freqs_GHz):
            S = S_per_freq[i]
            row = f'{fg}'
            for r in range(4):
                for c in range(4):
                    v = S[r, c]
                    row += f' {v.real} {v.imag}'
            f.write(row + '\n')


def _make_param(tmp_path, fb=28e9):
    p = SimpleNamespace()
    p.Z0 = 100.0
    p.fb = fb
    p.flim = float('inf')
    p.Txpskew = 0.0; p.Txnskew = 0.0; p.Rxpskew = 0.0; p.Rxnskew = 0.0
    p.snpPortsOrder = [1, 3, 2, 4]
    p.max_start_freq = 0.1e9
    p.FLAG = SimpleNamespace(S2P=False)
    p.package_testcase_i = 1
    return p


def _make_OP():
    return SimpleNamespace(
        INC_PACKAGE=0,
        RX_CALIBRATION=0,
        include_pcb=0,
        DISPLAY_WINDOW=True,
        ZERO_PAD=False,
        IDEAL_TX_TERM=False,
        IDEAL_RX_TERM=False,
        T_r_filter_type=0,
        PSDRXCAL=False,
    )


def _s4p_identity(freqs_GHz):
    """Identity S4P: S21=S12=1, all others=0."""
    S = np.zeros((4, 4), dtype=complex)
    S[1, 0] = 1.0; S[0, 1] = 1.0
    return [S.copy() for _ in freqs_GHz]


def _make_chdata(path):
    ch = SimpleNamespace()
    ch.filename = str(path)
    ch.ext = '.s4p'
    ch.type = 'THRU'
    ch.faxis = None
    return ch


def test_s4p_faxis_populated(tmp_path):
    """After read, ch.faxis is set to Hz."""
    freqs = np.array([1.0, 5.0, 10.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    assert chdata[0].faxis is not None
    assert chdata[0].faxis[0] == pytest.approx(1e9)
    assert chdata[0].faxis[-1] == pytest.approx(10e9)


def test_s4p_sdd21_raw_set(tmp_path):
    """sdd21_raw field is set after reading."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    assert hasattr(chdata[0], 'sdd21_raw')
    assert len(chdata[0].sdd21_raw) == 2


def test_s4p_orig_fields_present(tmp_path):
    """_orig copies exist for sdd21, sdd11, sdd12, sdd22."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    for field in ('sdd21_orig', 'sdd12_orig', 'sdd11_orig', 'sdd22_orig'):
        assert hasattr(chdata[0], field), f'Missing: {field}'


def test_inc_package_0_sdd21_equals_raw(tmp_path):
    """INC_PACKAGE=0: ch.sdd21 = sdd21_raw."""
    freqs = np.array([1.0, 5.0, 10.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()  # INC_PACKAGE=0
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    assert np.allclose(chdata[0].sdd21, chdata[0].sdd21_raw)


def test_unsupported_ext_raises(tmp_path):
    """Unsupported file extension raises ValueError."""
    ch = SimpleNamespace(filename='test.s3p', ext='.s3p', type='THRU', faxis=None)
    param = _make_param(tmp_path)
    OP = _make_OP()
    with pytest.raises(ValueError, match='unsupported extension'):
        read_s4p_files(param, OP, [ch])


def test_returns_four_outputs(tmp_path):
    """r4p15p0: read_s4p_files returns (chdata, SDDch, SDDp2p, param)."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    result = read_s4p_files(param, OP, [ch])
    assert len(result) == 4
    assert result[3] is param  # 4th output is param


def test_empty_portorder_auto_detects_and_persists(tmp_path):
    """r4p15p0: empty snpPortsOrder -> auto-detect; resolved order saved to param."""
    freqs = np.array([1.0, 5.0, 10.0, 20.0])
    S_per = []
    for _ in freqs:
        S = np.zeros((4, 4), dtype=complex)
        for d in range(4):
            S[d, d] = 0.05
        S[0, 1] = S[1, 0] = 0.9   # ports 1<->2
        S[2, 3] = S[3, 2] = 0.9   # ports 3<->4
        S_per.append(S)
    p = str(tmp_path / 'auto.s4p')
    _write_s4p(p, freqs, S_per)
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    param.snpPortsOrder = np.array([])  # empty -> auto
    OP = _make_OP()
    chdata, _, _, param_out = read_s4p_files(param, OP, [ch])
    assert list(param_out.snpPortsOrder) == [1, 3, 2, 4]
    assert chdata[0].faxis is not None


def test_bad_portorder_length_raises(tmp_path):
    """r4p15p0: non-empty port order whose length != 4 raises."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    param.snpPortsOrder = [1, 2, 3]  # length 3
    OP = _make_OP()
    with pytest.raises(ValueError, match='does not match'):
        read_s4p_files(param, OP, [ch])


def test_second_file_different_freq_raises(tmp_path):
    """Crosstalk file with different freq axis raises ValueError."""
    freqs1 = np.array([1.0, 5.0, 10.0])
    freqs2 = np.array([2.0, 6.0, 11.0])
    p1 = str(tmp_path / 'ch1.s4p')
    p2 = str(tmp_path / 'ch2.s4p')
    _write_s4p(p1, freqs1, _s4p_identity(freqs1))
    _write_s4p(p2, freqs2, _s4p_identity(freqs2))
    ch1 = _make_chdata(p1)
    ch2 = SimpleNamespace(filename=p2, ext='.s4p', type='FEXT', faxis=None)
    param = _make_param(tmp_path)
    OP = _make_OP()
    with pytest.raises(ValueError, match='frequency axis'):
        read_s4p_files(param, OP, [ch1, ch2])
