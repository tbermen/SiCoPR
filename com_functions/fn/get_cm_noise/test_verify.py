"""Verification tests for get_cm_noise().

# ============================================================
# MATLAB GROUND TRUTH
# For ki in 1..M: tps = PR(ki:M:end)  [1-based → Python ki-1::M]
# PR_fom = -testpdf.x(find(cdf_test>=BER,1,'first'))  when CM_norm_test=0
# results.CMn = best PR_fom over all ki
# results.CMn_p2p = max(PR) - min(PR)
#
# M=1: single sub-phase; tps=PR; testpdf built from PR; CMn computed once
# CM_norm_test=1: PR_fom = norm(tps); results.CMn = max over ki of norm(tps_ki)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_cm_noise.py_impl import get_cm_noise


def test_m1_norm_test():
    """M=1, CM_norm_test=1: CMn = norm(PR)."""
    PR = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
    op = SimpleNamespace(CM_norm_test=1)
    results = get_cm_noise(1, PR, 4, 1e-6, op)
    assert results.CMn == pytest.approx(float(np.linalg.norm(PR)), rel=1e-10)


def test_cmnp2p():
    """CMn_p2p = max(PR) - min(PR)."""
    PR = np.array([0.1, -0.3, 0.2])
    op = SimpleNamespace(CM_norm_test=1)
    results = get_cm_noise(1, PR, 4, 1e-6, op)
    assert results.CMn_p2p == pytest.approx(0.5)


def test_default_op():
    """OP=None: defaults to CM_norm_test=0; results has CMn and CMn_p2p."""
    PR = np.array([0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
    results = get_cm_noise(1, PR, 4, 1e-6)
    assert hasattr(results, 'CMn')
    assert hasattr(results, 'CMn_p2p')
    assert np.isfinite(results.CMn)


def test_op_missing_cm_norm_test():
    """OP without CM_norm_test field → defaults to 0."""
    PR = np.array([0.4, -0.4, 0.2, -0.2])
    op = SimpleNamespace()
    results = get_cm_noise(1, PR, 4, 1e-6, op)
    assert hasattr(results, 'CMn')


def test_multi_phase_picks_best():
    """M=2: selects the sub-phase with higher norm as best."""
    PR = np.array([0.1, 0.9, 0.1, 0.9])   # odd indices have large values
    op = SimpleNamespace(CM_norm_test=1)
    results = get_cm_noise(2, PR, 4, 1e-6, op)
    # sub-phase 0: PR[0::2]=[0.1,0.1]; sub-phase 1: PR[1::2]=[0.9,0.9]
    best_expected = float(np.linalg.norm([0.9, 0.9]))
    assert results.CMn == pytest.approx(best_expected, rel=1e-10)
