"""Tests for readdataSnPx (MATLAB lines ~10768).

MATLAB GROUND TRUTH:
  2-port Touchstone RI file → result.cs[1,0,:] swapped from file order.
  Frequency scaling: GHz in file → Hz in output.
  MA format: mag*exp(j*angle_deg*pi/180).
  DB format: 10^(db/20) * exp(j*angle_deg*pi/180).
"""
import os
import tempfile
import numpy as np
import pytest
from com_functions.fn.readdataSnPx.py_impl import readdataSnPx


def _write_s2p_ri(freq_ghz, s_data):
    """Write minimal RI-format s2p file. s_data shape: (nport, nport, nfreq)."""
    nport = s_data.shape[0]
    nfreq = len(freq_ghz)
    lines = ['! test\n', f'# GHz S RI R 50\n']
    for k in range(nfreq):
        row = [str(freq_ghz[k])]
        for i in range(nport):
            for j in range(nport):
                row.append(f'{s_data[i,j,k].real:.6f}')
                row.append(f'{s_data[i,j,k].imag:.6f}')
        lines.append(' '.join(row) + '\n')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.s2p', delete=False)
    tmp.writelines(lines)
    tmp.close()
    return tmp.name


def test_freq_scaling_ghz():
    freq_ghz = np.array([1.0, 2.0, 3.0])
    s = np.zeros((2, 2, 3), dtype=complex)
    f = _write_s2p_ri(freq_ghz, s)
    try:
        r = readdataSnPx(f, 2)
        np.testing.assert_allclose(r.freq, freq_ghz * 1e9)
    finally:
        os.unlink(f)


def test_2port_swap():
    freq_ghz = np.array([1.0])
    s = np.zeros((2, 2, 1), dtype=complex)
    s[0, 1, 0] = 0.3 + 0.1j  # S12
    s[1, 0, 0] = 0.5 + 0.2j  # S21 — will be swapped with S12
    f = _write_s2p_ri(freq_ghz, s)
    try:
        r = readdataSnPx(f, 2)
        # After swap: result.cs[0,1] should be what was originally S21=s[1,0]
        assert abs(r.cs[0, 1, 0] - (0.5 + 0.2j)) < 1e-5
        assert abs(r.cs[1, 0, 0] - (0.3 + 0.1j)) < 1e-5
    finally:
        os.unlink(f)


def test_output_shape():
    freq_ghz = np.linspace(1, 30, 10)
    nport = 4
    s = np.random.default_rng(0).standard_normal((nport, nport, 10)) + 1j * np.random.default_rng(1).standard_normal((nport, nport, 10))
    lines = ['! test\n', '# GHz S RI R 50\n']
    for k in range(10):
        row = [str(freq_ghz[k])]
        for i in range(nport):
            for j in range(nport):
                row.append(f'{s[i,j,k].real:.6f}')
                row.append(f'{s[i,j,k].imag:.6f}')
        lines.append(' '.join(row) + '\n')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.s4p', delete=False)
    tmp.writelines(lines)
    tmp.close()
    try:
        r = readdataSnPx(tmp.name, nport)
        assert r.cs.shape == (nport, nport, 10)
    finally:
        os.unlink(tmp.name)


def test_ri_values():
    freq_ghz = np.array([5.0])
    s = np.zeros((2, 2, 1), dtype=complex)
    s[0, 0, 0] = 0.1 + 0.2j
    f = _write_s2p_ri(freq_ghz, s)
    try:
        r = readdataSnPx(f, 2)
        assert abs(r.cs[0, 0, 0] - (0.1 + 0.2j)) < 1e-5
    finally:
        os.unlink(f)


def _write_s2p_ma(freq_ghz, mags, angles_deg):
    lines = ['! test\n', '# GHz S MA R 50\n']
    nport = 2
    for k in range(len(freq_ghz)):
        row = [str(freq_ghz[k])]
        for i in range(nport):
            for j in range(nport):
                row.append(f'{mags[i,j,k]:.6f}')
                row.append(f'{angles_deg[i,j,k]:.6f}')
        lines.append(' '.join(row) + '\n')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.s2p', delete=False)
    tmp.writelines(lines)
    tmp.close()
    return tmp.name


def test_ma_format():
    freq_ghz = np.array([2.0])
    mags = np.ones((2, 2, 1)) * 0.5
    angles = np.zeros((2, 2, 1))
    angles[0, 0, 0] = 90.0
    f = _write_s2p_ma(freq_ghz, mags, angles)
    try:
        r = readdataSnPx(f, 2)
        expected = 0.5 * np.exp(1j * np.pi / 2)
        assert abs(r.cs[0, 0, 0] - expected) < 1e-5
    finally:
        os.unlink(f)
