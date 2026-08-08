"""Verification tests for read_package_parameters().

# ============================================================
# MATLAB GROUND TRUTH (lines 10525-10593)
# Reads package parameters from a parameter block.
# mele=1/2/4: sets param.flex accordingly.
# mele=2: expands z_p_* to 4 columns with zeros.
# mele other: raises ValueError.
# ============================================================
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.read_package_parameters.py_impl import read_package_parameters
from types import SimpleNamespace


def _param_block_2col():
    """2-column flex package parameter block."""
    return {
        'C_p': np.array([0.1, 0.1]),
        'R_d': np.array([50.0, 50.0]),
        'A_v': np.array([0.5, 0.5]),
        'A_fe': np.array([0.3, 0.3]),
        'A_ne': np.array([0.2, 0.2]),
        'z_p (TX)': np.array([[5.0, 10.0]]),      # 1 case × 2 elements
        'z_p (NEXT)': np.array([[3.0, 6.0]]),
        'z_p (FEXT)': np.array([[4.0, 8.0]]),
        'z_p (RX)': np.array([[2.0, 4.0]]),
        'package_tl_gamma0_a1_a2': np.array([0.0, 1.734e-3, 1.455e-4]),
        'package_tl_tau': np.array([6.141e-3]),
        'package_Z_c': np.array([[78.2, 78.2]]),
    }


def _param_block_4col():
    return {
        'C_p': np.array([0.1, 0.1, 0.1, 0.1]),
        'R_d': np.array([50.0, 50.0, 50.0, 50.0]),
        'A_v': np.array([0.5, 0.5, 0.5, 0.5]),
        'A_fe': np.array([0.3, 0.3, 0.3, 0.3]),
        'A_ne': np.array([0.2, 0.2, 0.2, 0.2]),
        'z_p (TX)': np.array([[5.0, 10.0, 15.0, 20.0]]),
        'z_p (NEXT)': np.array([[3.0, 6.0, 9.0, 12.0]]),
        'z_p (FEXT)': np.array([[4.0, 8.0, 12.0, 16.0]]),
        'z_p (RX)': np.array([[2.0, 4.0, 6.0, 8.0]]),
        'package_tl_gamma0_a1_a2': np.array([0.0, 1.734e-3, 1.455e-4]),
        'package_tl_tau': np.array([6.141e-3]),
        'package_Z_c': np.array([[78.2, 78.2, 78.2, 78.2]]),
    }


def test_flex2_set():
    """mele=2 sets flex=2."""
    p = read_package_parameters(_param_block_2col())
    assert p.flex == 2


def test_flex4_set():
    """mele=4 sets flex=4."""
    p = read_package_parameters(_param_block_4col())
    assert p.flex == 4


def test_2col_expands_to_4():
    """mele=2 expands z_p_tx_cases to 4 columns."""
    p = read_package_parameters(_param_block_2col())
    assert p.z_p_tx_cases.shape[1] == 4


def test_invalid_mele_raises():
    """mele=3 raises ValueError."""
    block = _param_block_2col()
    block['z_p (TX)'] = np.array([[1.0, 2.0, 3.0]])
    block['z_p (NEXT)'] = np.array([[1.0, 2.0, 3.0]])
    block['z_p (FEXT)'] = np.array([[1.0, 2.0, 3.0]])
    block['z_p (RX)'] = np.array([[1.0, 2.0, 3.0]])
    block['package_Z_c'] = np.array([[78.2, 78.2, 78.2]])
    with pytest.raises(ValueError):
        read_package_parameters(block)


def test_returns_namespace():
    """Returns a SimpleNamespace."""
    p = read_package_parameters(_param_block_2col())
    assert hasattr(p, 'z_p_tx_cases')
    assert hasattr(p, 'R_diepad')


def test_appends_to_existing():
    """Can append to an existing param_struct."""
    existing = SimpleNamespace(other_field=42)
    p = read_package_parameters(_param_block_2col(), existing)
    assert p.flex == 2
    assert p.other_field == 42
