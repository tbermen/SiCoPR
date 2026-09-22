"""Verification tests for PRBS13Q().

# ============================================================
# MATLAB GROUND TRUTH (lines 4174-4192)
# LFSR with taps=[13,12,2,1], seed=[0,0,0,0,0,1,0,1,0,1,0,1,1]
# generates 2^13-2 = 8190 NRZ bits.
# PAM4 mapping pairs NRZ bits → {-1, -1/3, 1/3, 1} (Grey coding)
# seq has 4095 PAM4 samples
# syms: integers {0,1,2,3}
# syms_nrz: NRZ bit sequence
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.PRBS13Q.py_impl import PRBS13Q


def test_returns_three_outputs():
    seq, syms, syms_nrz = PRBS13Q()
    assert seq is not None
    assert syms is not None
    assert syms_nrz is not None


def test_seq_values_are_pam4():
    """seq contains only {-1, -1/3, 1/3, 1}."""
    seq, _, _ = PRBS13Q()
    valid = {-1.0, -1.0/3.0, 1.0/3.0, 1.0}
    for v in seq:
        assert any(abs(v - x) < 1e-10 for x in valid), f'unexpected value {v}'


def test_syms_integer_range():
    """syms contains integers in {0, 1, 2, 3}."""
    _, syms, _ = PRBS13Q()
    assert np.all((syms >= 0) & (syms <= 3))


def test_seq_length():
    """seq has 4095 samples (2^13-2 bits → 4095 PAM4 symbols)."""
    seq, _, _ = PRBS13Q()
    assert len(seq) == 4095


def test_syms_nrz_is_binary():
    """syms_nrz values are +1 or -1."""
    _, _, syms_nrz = PRBS13Q()
    unique = np.unique(syms_nrz)
    assert set(unique.tolist()) == {-1.0, 1.0}


def test_all_four_levels_present():
    """All four PAM4 levels appear in seq."""
    seq, _, _ = PRBS13Q()
    levels = {-1.0, -1.0/3.0, 1.0/3.0, 1.0}
    for lvl in levels:
        assert any(abs(v - lvl) < 1e-10 for v in seq), f'level {lvl} not found'


# ============================================================
# COM Octave oracle values (2026-09-22)
# PRBS13Q run verbatim from octave/com_ieee8023_4p16p0_octave_compat.m via
# tools/octave_oracle.py, with LFSR and pam as its subfunctions. The compat
# bodies of all three are byte-identical to matlab/com_ieee8023_4p16p0.m.
#
# All three outputs matched the Python port ELEMENTWISE AND EXACTLY --
# seq (4095), syms (4095) and syms_nrz (8191) -- so the values below are the
# reference's, not the port's, and they are pinned to hold it there.
#
# Note the odd length: LFSR returns c(:,n)' with 2^13-1 = 8191 rows, so
# syms_nrz is 8191 long and pam drops the last, unpaired bit. The 8190 in the
# header comment above is the loop count, not the output length.
# ============================================================

import hashlib   # noqa: E402

_OCT_SYMS_SHA256 = '7b18e164d4d8b13c80c5db3955f86dbb1b097a24ffc937585d755f9b7b2ad81f'
_OCT_SYMS_FIRST32 = '21110003102131230333200310232202'
_OCT_SYMS_LAST32 = '22131021012303013322120300323132'
_OCT_LEVEL_COUNTS = [1014, 1002, 1014, 1065]
_OCT_NRZ_FIRST16 = [1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1]
_OCT_NRZ_LAST16 = [-1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1]
_OCT_SEQ_FIRST8 = [0.3333333333333333, -0.3333333333333333,
                   -0.3333333333333333, -0.3333333333333333,
                   -1.0, -1.0, -1.0, 1.0]
_OCT_SEQ_LAST4 = [1.0, -0.3333333333333333, 1.0, 0.3333333333333333]


def test_octave_output_lengths():
    """COM Octave: size(seq)=1x4095, size(syms)=1x4095, size(syms_nrz)=1x8191."""
    seq, syms, syms_nrz = PRBS13Q()
    assert len(seq) == 4095
    assert len(syms) == 4095
    assert len(syms_nrz) == 8191


def test_octave_syms_sequence_exact():
    """COM Octave: the whole 4095-symbol sequence, pinned by its digits."""
    _, syms, _ = PRBS13Q()
    digits = ''.join(str(int(v)) for v in syms)
    assert digits[:32] == _OCT_SYMS_FIRST32
    assert digits[-32:] == _OCT_SYMS_LAST32
    assert hashlib.sha256(digits.encode()).hexdigest() == _OCT_SYMS_SHA256


def test_octave_level_counts():
    """COM Octave: syms==0,1,2,3 appear 1014, 1002, 1014, 1065 times."""
    _, syms, _ = PRBS13Q()
    syms = np.asarray(syms)
    assert [int((syms == v).sum()) for v in (0, 1, 2, 3)] == _OCT_LEVEL_COUNTS


def test_octave_seq_values_exact():
    """COM Octave: seq(1:8) and seq(end-3:end), to the last bit."""
    seq, _, _ = PRBS13Q()
    seq = np.asarray(seq, dtype=float)
    assert list(seq[:8]) == _OCT_SEQ_FIRST8
    assert list(seq[-4:]) == _OCT_SEQ_LAST4


def test_octave_syms_nrz_edges():
    """COM Octave: syms_nrz(1:16), syms_nrz(end-15:end) and sum = 1."""
    _, _, syms_nrz = PRBS13Q()
    syms_nrz = np.asarray(syms_nrz, dtype=float)
    assert [int(v) for v in syms_nrz[:16]] == _OCT_NRZ_FIRST16
    assert [int(v) for v in syms_nrz[-16:]] == _OCT_NRZ_LAST16
    assert float(syms_nrz.sum()) == 1.0
