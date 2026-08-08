"""Verification tests for get_sigma_noise().

# ============================================================
# MATLAB GROUND TRUTH
# sigma_NE = sigma_bn * sqrt(mean(abs(H_np(1:idxfbby2)).^2))
# sigma_HP = sigma_bn * mean(abs(H_hp(1:idxfbby2)).^2)
# H_np = Hnoise_channel .* H_ctf .* H_r .* H_hp
#
# With f_hp=0: H_hp = 1 → H_np = Hnoise_channel .* H_ctf .* H_r
# With H_ctf=1, Hnoise=1, H_r=1, H_hp=1: sigma_NE = sigma_bn*1 = sigma_bn
#
# sigma_HP <= sigma_NE * sigma_NE / sigma_bn because H_np includes more terms
# Both outputs are real scalars.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_sigma_noise.py_impl import get_sigma_noise


def make_chdata(n=50, fb=100e9, f_r=0.75):
    faxis = np.linspace(0, fb, n)
    sdd21 = np.ones(n, dtype=complex)
    ch = SimpleNamespace(faxis=faxis, sdd21=sdd21)
    return [ch, ch]   # chdata[0] and chdata[1]


def make_param(fb=100e9, f_r=0.75, f_hp=0.0):
    return SimpleNamespace(fb=fb, f_r=f_r, f_hp=f_hp)


def test_returns_two_scalars():
    chdata = make_chdata()
    p = make_param()
    out = get_sigma_noise(np.ones(50), p, chdata, 0.01)
    assert len(out) == 2
    assert np.isscalar(out[0]) or out[0].ndim == 0
    assert np.isscalar(out[1]) or out[1].ndim == 0


def test_proportional_to_sigma_bn():
    """Both outputs scale linearly with sigma_bn."""
    chdata = make_chdata()
    p = make_param()
    H_ctf = np.ones(50, dtype=complex)
    ne1, hp1 = get_sigma_noise(H_ctf, p, chdata, 1.0)
    ne2, hp2 = get_sigma_noise(H_ctf, p, chdata, 2.0)
    assert ne2 == pytest.approx(2 * ne1, rel=1e-10)
    assert hp2 == pytest.approx(2 * hp1, rel=1e-10)


def test_fhp_zero_no_hp_filter():
    """f_hp=0: H_hp=1, no high-pass filtering."""
    chdata = make_chdata()
    p = make_param(f_hp=0.0)
    ne, hp = get_sigma_noise(np.ones(50), p, chdata, 1.0)
    assert np.isfinite(ne)
    assert np.isfinite(hp)


def test_non_negative_outputs():
    """sigma_NE and sigma_HP must be non-negative."""
    chdata = make_chdata()
    p = make_param(f_hp=10e9)
    ne, hp = get_sigma_noise(np.ones(50), p, chdata, 0.05)
    assert ne >= 0
    assert hp >= 0
