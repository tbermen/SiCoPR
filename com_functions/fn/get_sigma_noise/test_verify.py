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


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.  Probe inputs (built by _probe_inputs below):
#   n = 51, fb = 100e9, f_r = 0.75, sigma_bn = 0.01
#   faxis = linspace(0, fb, 51)              (column, as chdata stores it)
#   sdd21 = 0.9.^(1:51) .* exp(-1i*0.03*(0:50))
#   H_ctf = 0.8 + 0.01i*(0:50)               (column, matching faxis)
#
# idxfbby2 = find(faxis >= fb/2, 1) is EMPTY when the axis stops short:
#   faxis = linspace(0, fb/4, 51) -> H_np(1:[]) is empty, mean([]) is NaN, so
#     sigma_NE = NaN and sigma_HP = NaN.
#   np.argmax answers 0 on an all-False mask, so the old code averaged over
#   the first bin alone and returned (0, 0) for this data — a finite number
#   with nothing behind it.
#
# abs(H(1:n).^2) squares before taking the modulus.  |z^2| and |z|^2 are
# equal in exact arithmetic but not in doubles — 18 of the 26 in-band
# elements differ — so the reference order is kept.  With f_hp = 0 that makes
# sigma_NE bit-identical to Octave:
#   f_hp = 0   -> sigma_NE = 0.0032415691244300788, sigma_HP = 0.01
#                 (the abs(z)^2 form gave 0.0032415691244300792)
#   f_hp = 1e9 -> sigma_NE = 0.0028380346912153954, sigma_HP =
#                 0.0094820988935225836.  Python lands 1-2 ulp away here;
#                 that residue is Octave-vs-numpy complex arithmetic, not
#                 this function: H_r = 1./polyval(...) already differs by up
#                 to 1.3e-15 relative before any of it is used.
#
# MEASURED DIVERGENCE, not fixed here: orientation.  chdata(2).faxis is a
# column in COM and H_ctf is derived from it, so both agree in the engine.
# If a caller passes H_ctf as a ROW against a column faxis, MATLAB implicit-
# expands H_np into a 51x51 matrix and H_np(1:idxfbby2) then reads down the
# FIRST COLUMN, giving sigma_NE = 0.0032323013154612157 for the f_hp = 0
# probe against 0.0032415691244300788 for the elementwise form — a 0.3%
# difference, not a rounding one.  Python ravels both and always computes the
# elementwise product.  Reproducing implicit expansion would mean carrying
# 2-D shapes through the whole port, so it is recorded rather than emulated.
# ============================================================


def _probe_inputs(fmax=100e9, n=51):
    fb = 100e9
    faxis = np.linspace(0.0, fmax, n)
    sdd21 = (0.9 ** (1 + np.arange(n))) * np.exp(-1j * 0.03 * np.arange(n))
    H_ctf = 0.8 * np.ones(n, dtype=complex) + 0.01j * np.arange(n)
    ch = SimpleNamespace(faxis=faxis, sdd21=sdd21)
    return H_ctf, SimpleNamespace(fb=fb, f_r=0.75, f_hp=0.0), [ch, ch]


def test_no_faxis_above_half_fb_is_nan():
    """find() is empty, so the reference averages an empty slice: NaN, not 0."""
    H_ctf, p, chdata = _probe_inputs(fmax=100e9 / 4)
    p.f_hp = 1e9
    ne, hp = get_sigma_noise(H_ctf, p, chdata, 0.01)
    assert np.isnan(ne)
    assert np.isnan(hp)


def test_fhp_zero_matches_octave_exactly():
    """abs(z.^2), not abs(z).^2 — the two differ in the last bit."""
    H_ctf, p, chdata = _probe_inputs()
    ne, hp = get_sigma_noise(H_ctf, p, chdata, 0.01)
    assert ne == 0.0032415691244300788
    assert hp == 0.01


def test_fhp_nonzero_matches_octave():
    """1-2 ulp of Octave; the residue is complex-arithmetic, not algorithmic."""
    H_ctf, p, chdata = _probe_inputs()
    p.f_hp = 1e9
    ne, hp = get_sigma_noise(H_ctf, p, chdata, 0.01)
    assert ne == pytest.approx(0.0028380346912153954, rel=1e-15)
    assert hp == pytest.approx(0.0094820988935225836, rel=1e-15)


def test_faxis_starting_at_half_fb():
    """idxfbby2 = 1: only the first bin is in band."""
    H_ctf, p, chdata = _probe_inputs()
    ch = chdata[0]
    ch.faxis = np.linspace(100e9 / 2, 100e9, len(ch.faxis))
    p.f_hp = 1e9
    ne, hp = get_sigma_noise(H_ctf, p, [ch, ch], 0.01)
    assert ne == pytest.approx(0.0070621020704907805, rel=1e-14)
    assert hp == pytest.approx(0.0099960015993602568, rel=1e-14)
