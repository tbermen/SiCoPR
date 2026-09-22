"""Verification tests for TD_FD_fillin().

# ============================================================
# MATLAB GROUND TRUTH (lines 4610-4674)
# Computes frequency-domain fields from time-domain pulse response.
# IL_conv = fd[:f75] / (Vf*M*2) / prr / H_ftr.
# IL fields are set on all chdata channels.
# SDDch, SDDp2p initialised from first channel.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.TD_FD_fillin.py_impl import TD_FD_fillin


def _param(M=8, fb=25e9):
    return SimpleNamespace(
        fb=fb,
        ui=1.0 / fb,
        samples_per_ui=M,
        N_v=10,
        BTorder=2,
        fb_BT_cutoff=1.0,
        fb_BW_cutoff=1.0,
    )


def _op():
    return SimpleNamespace(Bessel_Thomson=False, Butterworth=False)


def _chdata(M=8, fb=25e9, N=2048):
    # N must be at least 200*samples_per_ui: the reference indexes
    # T(1:200*M) when it builds its (unused) STEP timeseries, and errors
    # below that -- see the oracle block at the end of this file.
    dt = 1.0 / (fb * M)
    T = np.arange(N) * dt
    ir = np.zeros(N)
    ir[0] = 1.0  # unit impulse at t=0
    pulse = np.cumsum(ir)
    cd = SimpleNamespace(
        uneq_pulse_response=pulse,
        t=T,
        type='THRU',
    )
    return [cd]


def test_returns_four_outputs():
    """TD_FD_fillin returns (chdata, param, SDDch, SDDp2p)."""
    result = TD_FD_fillin(_param(), _op(), _chdata())
    assert len(result) == 4


def test_il_fields_set():
    """sdd21_raw field is set on chdata[0]."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _chdata())
    assert hasattr(chdata[0], 'sdd21_raw')


def test_faxis_set():
    """faxis is set on chdata[0]."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _chdata())
    assert hasattr(chdata[0], 'faxis')
    assert chdata[0].faxis[0] == pytest.approx(0.0)


def test_sddch_shape():
    """SDDch has shape (n_freq, 2, 2)."""
    _, _, SDDch, _ = TD_FD_fillin(_param(), _op(), _chdata())
    assert SDDch.ndim == 3
    assert SDDch.shape[1] == 2
    assert SDDch.shape[2] == 2


def test_zero_fields_are_zero():
    """sdd11_raw (zero field) is all zeros."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _chdata())
    assert np.all(chdata[0].sdd11_raw == 0)


# ============================================================
# COM Octave 4p16p0 oracle pins.
# TD_FD_fillin extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m), with `timeseries` stubbed out -- Octave has
# no such class, and its result STEP is never read.
#
# Fixture: fb=25e9, M=8, 2048 samples, t = (0:2047)/(fb*M), and
# uneq_pulse_response = cumsum(exp(-((n-20)/4).^2)) as a COLUMN.  The column
# matters in the reference: prr and H_ftr are forced to columns with (:), so
# a row pulse response broadcasts fd against them and IL_conv comes back as
# an N-by-N matrix instead of a vector.  The port is orientation-agnostic.
#
# Not matched, and reported rather than patched: Octave's colon operator
# snaps the last element of f to its limit 1/dt, where np.arange computes
# 2047*df; the two differ by one ulp (200000000000 vs 200000000000.00003) at
# f(end) only, every other bin agreeing bit for bit.
# ============================================================
_OCT_N_IL = 193
_OCT_DF = 97703957.010258928
_OCT_F_191_194 = [18661455788.959454, 18759159745.969715, 18856863702.979973]

_OCT_IL_0_3 = [0.64031937534083827 + 0j,
               -0.0063110440424757937 + 0.00018786603988831064j,
               -0.0062998243859800195 + 0.00037538945027520645j]
_OCT_IL_95_98 = [0.00035399166428162201 + 0.00055412671182266532j,
                 0.00029730859887983925 + 0.00052548044794109149j,
                 0.00024016403712663444 + 0.0005006059948687952j]
_OCT_IL_190_193 = [-8.497323151181936e-05 + 0.0015317504898524935j,
                   -0.0001053504819597691 + 0.0015268458164375469j,
                   -0.0001273529249326428 + 0.0015234250271638661j]
_OCT_IL_SUM_ABS = 1.161934443027814

_OCT_ILF_0_3 = [0.64031937534083827 + 0j,
                -0.0063130836417566267 + 9.8734227940633416e-05j,
                -0.0063079766233718323 + 0.00019734887082615996j]
_OCT_ILF_95_98 = [-0.00047975644372841032 + 0.00047238039634679908j,
                  -0.00046937384511602488 + 0.00040282330813414967j,
                  -0.00046091846039493251 + 0.00033384536746717148j]
_OCT_ILF_190_193 = [-0.00033526161653017869 - 0.001737237131177008j,
                    -0.00028049333214930609 - 0.0017478701028017588j,
                    -0.00022377282663306794 - 0.0017593058310134042j]
_OCT_ILF_SUM_ABS = 1.1747491857669976


def _smooth_chdata(M=8, fb=25e9, N=2048):
    dt = 1.0 / (fb * M)
    n = np.arange(N)
    pulse = np.cumsum(np.exp(-((n - 20.0) / 4.0) ** 2))
    return [SimpleNamespace(uneq_pulse_response=pulse, t=n * dt, type='THRU')]


def test_octave_frequency_axis():
    """COM Octave: f = 0:1/max(T):1/dt, so df = 1/(2047*dt) and fmaxi is the
    number of bins."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _smooth_chdata())
    f = np.asarray(chdata[0].faxis).ravel()
    assert len(f) == 2048
    assert chdata[0].fmaxi == 2048
    assert f[0] == 0.0
    assert f[1] == pytest.approx(_OCT_DF, rel=1e-15)
    np.testing.assert_allclose(f[191:194], _OCT_F_191_194, rtol=1e-15)


def test_octave_IL_conv_keeps_the_first_bin_at_or_above_three_quarter_baud():
    """COM Octave: f75 is 1-based, so fd(1:f75) keeps the crossing bin too.

    f(193) = 18759159745.969715 is the first bin at or above 0.75*fb, and
    length(IL_conv) is 193.  Slicing at the 0-based index gave 192 bins, and
    with them a short sdd21 and a short SDDch.
    """
    chdata, _, SDDch, SDDp2p = TD_FD_fillin(_param(), _op(), _smooth_chdata())
    IL = np.asarray(chdata[0].sdd21_raw).ravel()
    assert len(IL) == _OCT_N_IL
    assert SDDch.shape == (_OCT_N_IL, 2, 2)
    assert len(np.asarray(SDDp2p).ravel()) == _OCT_N_IL
    f = np.asarray(chdata[0].faxis).ravel()
    assert f[_OCT_N_IL - 2] < _param().fb * 0.75 <= f[_OCT_N_IL - 1]


def test_octave_IL_conv_values_unfiltered():
    """COM Octave, Bessel_Thomson and Butterworth both off."""
    chdata, *_ = TD_FD_fillin(_param(), _op(), _smooth_chdata())
    IL = np.asarray(chdata[0].sdd21_raw).ravel()
    np.testing.assert_allclose(IL[0:3], _OCT_IL_0_3, rtol=1e-12)
    np.testing.assert_allclose(IL[95:98], _OCT_IL_95_98, rtol=1e-12)
    np.testing.assert_allclose(IL[190:193], _OCT_IL_190_193, rtol=1e-12)
    assert float(np.sum(np.abs(IL))) == pytest.approx(_OCT_IL_SUM_ABS,
                                                      rel=1e-12)


def test_octave_IL_conv_values_with_both_filters():
    """COM Octave, H_ftr = Butterworth * Bessel-Thomson, BTorder 2 and both
    cutoffs at fb."""
    op = _op()
    op.Bessel_Thomson = True
    op.Butterworth = True
    chdata, *_ = TD_FD_fillin(_param(), op, _smooth_chdata())
    IL = np.asarray(chdata[0].sdd21_raw).ravel()
    np.testing.assert_allclose(IL[0:3], _OCT_ILF_0_3, rtol=1e-12)
    np.testing.assert_allclose(IL[95:98], _OCT_ILF_95_98, rtol=1e-12)
    np.testing.assert_allclose(IL[190:193], _OCT_ILF_190_193, rtol=1e-12)
    assert float(np.sum(np.abs(IL))) == pytest.approx(_OCT_ILF_SUM_ABS,
                                                      rel=1e-12)


def test_octave_short_time_axis_is_refused():
    """COM Octave: the reference evaluates T(1:200*samples_per_ui) for its
    unused STEP timeseries, so 1599 samples at M=8 errors
    'T(1600): out of bound 1599' while 1600 answers."""
    with pytest.raises(IndexError):
        TD_FD_fillin(_param(), _op(), _smooth_chdata(N=1599))
    chdata, *_ = TD_FD_fillin(_param(), _op(), _smooth_chdata(N=1600))
    assert len(np.asarray(chdata[0].sdd21_raw).ravel()) == 151


def test_octave_every_channel_is_filled_and_SDDch_comes_from_the_first():
    """COM Octave: the IL fields carry IL_conv and the zero fields are zero on
    every channel, while SDDch and SDDp2p are built once, at i==1."""
    cd = _smooth_chdata()
    cd.append(SimpleNamespace(uneq_pulse_response=0.3 * cd[0].uneq_pulse_response,
                              t=cd[0].t, type='NEXT'))
    chdata, _, SDDch, _ = TD_FD_fillin(_param(), _op(), cd)
    for ch in chdata:
        for field in ('sdd12_raw', 'sdd21_raw', 'sdd12_orig', 'sdd21_orig',
                      'sdd12', 'sdd21', 'sdd21p', 'sdd21f'):
            assert len(np.asarray(getattr(ch, field)).ravel()) == _OCT_N_IL
        for field in ('sdd11_raw', 'sdd22_raw', 'sdc11_raw', 'sdc21_raw'):
            assert np.all(np.asarray(getattr(ch, field)) == 0)
        assert ch.TX_RL == [] and ch.TDR11 == [] and ch.PDTR22 == []
    np.testing.assert_allclose(SDDch[:, 1, 0],
                               np.asarray(chdata[0].sdd21_raw).ravel(),
                               rtol=1e-15)
    assert np.all(SDDch[:, 0, 0] == 0)
    assert np.all(SDDch[:, 1, 1] == 0)
