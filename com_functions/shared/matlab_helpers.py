"""Shared Python utilities for COM MATLAB-to-Python conversion."""
from types import SimpleNamespace
import numpy as np


def matlab_struct(**kwargs):
    """Create a SimpleNamespace (MATLAB struct equivalent)."""
    return SimpleNamespace(**kwargs)


def isempty(x):
    """MATLAB isempty equivalent."""
    if x is None:
        return True
    arr = np.asarray(x)
    return arr.size == 0


def matlab_length(x):
    """MATLAB length(x) = max(size(x))."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return 1
    return max(arr.shape)
