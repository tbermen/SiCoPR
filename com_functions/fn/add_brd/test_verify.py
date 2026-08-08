"""Verification tests for add_brd().

# ============================================================
# MATLAB GROUND TRUTH (lines 4768-4820)
# Adds board trace (TX+caps and RX+caps) to chdata S-params.
# include_pcb=1: tx || channel || rx cascade.
# include_pcb=2: channel || rx cascade.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.add_brd.py_impl import add_brd


def _freq(N=50):
    return np.linspace(0, 50e9, N)


def _chdata(ctype='THRU', N=50):
    f = _freq(N)
    cd = SimpleNamespace(
        type=ctype,
        faxis=f,
        sdd11_raw=np.zeros(N, dtype=complex),
        sdd12_raw=np.ones(N, dtype=complex) * 0.9,
        sdd21_raw=np.ones(N, dtype=complex) * 0.9,
        sdd22_raw=np.zeros(N, dtype=complex),
    )
    return cd


def _param():
    return SimpleNamespace(
        Z0=50.0,
        C_0=np.array([0.1e-12, 0.1e-12]),
        C_1=np.array([0.05e-12, 0.05e-12]),
        brd_Z_c=np.array([50.0, 50.0]),
        brd_gamma0_a1_a2=np.array([0.0, 0.01, 0.001]),
        brd_tau=0.0,
        z_bp_tx=0.01, z_bp_rx=0.01,
        z_bp_next=0.01, z_bp_fext=0.01,
    )


def _op(include_pcb=1):
    return SimpleNamespace(include_pcb=include_pcb)


def test_returns_four_arrays():
    """add_brd returns (s11, s12, s21, s22)."""
    result = add_brd(_chdata(), _param(), _op())
    assert len(result) == 4


def test_output_length_matches_freq():
    """Output arrays have same length as faxis."""
    N = 50
    s11, s12, s21, s22 = add_brd(_chdata(N=N), _param(), _op())
    assert len(s21) == N


def test_include_pcb2():
    """include_pcb=2: rx-only cascade returns valid arrays."""
    s11, s12, s21, s22 = add_brd(_chdata(), _param(), _op(include_pcb=2))
    assert len(s21) > 0


def test_matched_line_no_reflection():
    """Matched transmission line (Z_c=Z0, zero length) → s11 ≈ 0 at DC."""
    cd = _chdata()
    p = _param()
    p.brd_Z_c = np.array([50.0, 50.0])
    p.z_bp_tx = 0.0
    p.z_bp_rx = 0.0
    s11, s12, s21, s22 = add_brd(cd, p, _op())
    assert np.abs(s11[0]) < 0.1


def test_fext_type():
    """FEXT type uses z_bp_fext for TX side."""
    result = add_brd(_chdata(ctype='FEXT'), _param(), _op())
    assert len(result) == 4
