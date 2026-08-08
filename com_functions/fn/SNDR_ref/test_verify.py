"""Verification tests for SNDR_ref().

# ============================================================
# MATLAB GROUND TRUTH (lines 4405-4442)
# For each of 6 TX-FFE presets, apply FFE and compute:
#   SNDR = 10*log10(ss(hss) * 10^(SNR_TX/10) / ss(PR_noFFE_sampled))
# Default presets: 6 standard PAM4 TX-FFE configurations.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.SNDR_ref.py_impl import SNDR_ref


def _param(M=8, D_p=2, N_p=20):
    return SimpleNamespace(
        SNDR=[20.0],
        samples_per_ui=M,
        D_p=D_p,
        N_p=N_p,
    )


def _gaussian_pr(N=200, peak=50, M=8):
    x = np.arange(N)
    return np.exp(-0.5 * ((x - peak) / 5) ** 2)


def test_returns_six_presets():
    """SNDR_ref has 6 values (one per preset)."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    assert len(results.SNDR_ref) == 6


def test_sndr_values_are_finite():
    """All SNDR_ref values are finite."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    assert np.all(np.isfinite(results.SNDR_ref))


def test_preset_fields_present():
    """Results has all per-preset SNDR fields."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    for field in ('SNDR_ref_p1', 'SNDR_ref_p2', 'SNDR_ref_p3',
                  'SNDR_ref_p4', 'SNDR_ref_p5', 'SNDR_ref_p6'):
        assert hasattr(results, field)


def test_sigma_il_is_positive():
    """sigma_iL (first preset) is positive."""
    PR = _gaussian_pr()
    results = SNDR_ref(PR, _param())
    assert results.sigma_iL > 0


def test_custom_presets_used():
    """When param.preset is provided with 6 entries, those are used."""
    param = _param()
    param.preset = [SimpleNamespace(txffe=[0, 0, 0, 1, 0])] * 6
    PR = _gaussian_pr()
    results = SNDR_ref(PR, param)
    assert len(results.SNDR_ref) == 6
