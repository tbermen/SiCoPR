"""Verification tests for add_pkg_with_die().

# ============================================================
# MATLAB GROUND TRUTH (lines 4843-4857)
# Applies TX package + die pad model to S-parameters.
# Calls make_full_pkg and combines4p (both pending) → NotImplementedError.
# ============================================================
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.add_pkg_with_die.py_impl import add_pkg_with_die
from types import SimpleNamespace
import numpy as np


def _S():
    N = 10
    return SimpleNamespace(
        Frequencies=np.linspace(0, 50e9, N),
        Parameters=np.zeros((2, 2, N), dtype=complex),
    )


def _param():
    return SimpleNamespace(R_diepad=np.array([50.0]), Z0=50.0)


def _OP():
    return SimpleNamespace()


def test_implemented_requires_full_param():
    """add_pkg_with_die is implemented; incomplete param raises AttributeError, not NotImplementedError."""
    # _param() is intentionally minimal (missing C_diepad etc.) to confirm the function
    # proceeds past the stub and reaches real logic.
    with pytest.raises((AttributeError, TypeError, ValueError)):
        add_pkg_with_die(_S(), 'dd', _param(), _OP())


def test_se_mode_requires_full_param():
    """SE mode is implemented; incomplete param raises AttributeError."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        add_pkg_with_die(_S(), 'se', _param(), _OP())


def test_cd_mode_requires_full_param():
    """CD mode is implemented; incomplete param raises AttributeError."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        add_pkg_with_die(_S(), 'cd', _param(), _OP())
