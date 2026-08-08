"""Verification tests for add_brdorig().

# ============================================================
# MATLAB GROUND TRUTH (lines 4821-4841)
# switch chdata.type: 'THRU'→z_bp_tx; 'NEXT'→z_bp_next; 'FEXT'→z_bp_fext
# synth_tline for TX and RX board traces
# include_pcb=1: tx→ch→rx cascade (three stages)
# include_pcb=2: ch→rx cascade (two stages)
# Returns (s11, s12, s21, s22) as 1-D arrays of length N.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.add_brdorig.py_impl import add_brdorig


def _param():
    return SimpleNamespace(
        z_bp_tx=0.05, z_bp_next=0.04, z_bp_fext=0.03, z_bp_rx=0.06,
        brd_Z_c=np.array([50.0, 50.0]),
        Z0=50.0,
        brd_gamma0_a1_a2=np.array([0.0, 0.01, 0.001]),
        brd_tau=0.0,
    )


def _chdata(chtype='THRU', N=10):
    f = np.linspace(1e9, 20e9, N)
    s0 = np.zeros(N, dtype=complex)
    s1 = np.ones(N, dtype=complex)
    return SimpleNamespace(
        type=chtype, faxis=f,
        sdd11_raw=s0, sdd12_raw=s1, sdd21_raw=s1, sdd22_raw=s0,
    )


def _op(include_pcb=1):
    return SimpleNamespace(include_pcb=include_pcb)


def test_thru_pcb1_output_length():
    """THRU + include_pcb=1: four 1-D arrays of length N."""
    N = 15
    s11, s12, s21, s22 = add_brdorig(_chdata('THRU', N), _param(), _op(1))
    for arr in (s11, s12, s21, s22):
        assert len(arr) == N


def test_next_pcb2_output_length():
    """NEXT + include_pcb=2: uses z_bp_next, still correct length."""
    N = 10
    s11, s12, s21, s22 = add_brdorig(_chdata('NEXT', N), _param(), _op(2))
    assert len(s21) == N


def test_fext_uses_z_bp_fext():
    """FEXT type selects z_bp_fext; no error."""
    N = 8
    s11, s12, s21, s22 = add_brdorig(_chdata('FEXT', N), _param(), _op(1))
    assert len(s21) == N


def test_unknown_type_raises():
    """Unknown chdata.type raises ValueError."""
    with pytest.raises(ValueError, match='Unknown chdata.type'):
        add_brdorig(_chdata('XMIT'), _param(), _op(1))


def test_pcb1_vs_pcb2_differ():
    """include_pcb=1 and =2 produce different S21 (different cascade stages)."""
    cd = _chdata('THRU')
    _, _, s21_1, _ = add_brdorig(cd, _param(), _op(1))
    _, _, s21_2, _ = add_brdorig(cd, _param(), _op(2))
    assert not np.allclose(s21_1, s21_2)
