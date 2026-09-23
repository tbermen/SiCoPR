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
from com_functions.fn.PRBS13Q.py_impl import PRBS13Q, _lfsr


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


# --------------------------------------------------------------------------
# LFSR on its own, against COM Octave.
#
# PRBS13Q calls LFSR exactly once, with taps [13 12 2 1] and one seed, so the
# 8191-symbol sequence pinned above covers that single argument and nothing
# else. LFSR is the reference's own function (ML 4195-4211); the assembler
# inlines it here as _lfsr because it has no other caller, and an inlined
# helper with no test of its own is a gap, not an exemption.
#
# What is worth driving directly is register indexing. The shift is a
# VECTORISED MATLAB assignment, s(n+1-j) = s(n-j) for j = 1:n-1, where every
# element on the right is read before any element on the left is written. The
# port writes it as a descending loop, which is equivalent; an ascending loop
# would smear s(1) through the whole register and is caught here. The tap loop
# bound moves with the number of taps, and the m == 2 case skips it entirely,
# so the cases below run 2, 3, 4 and 5 taps, taps given out of order, and an
# all-ones seed.
#
# COM Octave 4p16p0, 2026-09-23. seq is pinned by length, popcount and its
# leading bits, and the state history c by its shape and total, which together
# fix the whole register trajectory without carrying thousands of literals.
# --------------------------------------------------------------------------

# case, seed, taps, len(seq), sum(seq), first bits of seq, sum(c)
_LFSR_OCT = (
    ('2 taps, n=4', [1, 0, 0, 1], [4, 3],
     15, 8, '100110101111000', 32),
    ('2 taps, n=5', [0, 1, 1, 0, 1], [5, 3],
     31, 16, '1011001111100011011101010000100', 80),
    ('3 taps, n=5', [1, 1, 0, 1, 0], [5, 4, 2],
     31, 16, '0101100100011110101100100011110', 83),
    ('4 taps, n=6', [0, 0, 1, 0, 1, 1], [6, 5, 3, 1],
     63, 42, '11010010111111010010111111010010', 248),
    ('5 taps, n=7', [1, 0, 1, 1, 0, 0, 1], [7, 6, 5, 2, 1],
     127, 52, '10011010000010110011010000010110', 354),
    ('taps out of order, n=6', [1, 1, 1, 0, 0, 1], [2, 6, 1, 4],
     63, 36, '10011111101100010001110011111101', 216),
    ('all-ones seed, n=5', [1, 1, 1, 1, 1], [5, 4, 3, 1],
     31, 16, '1111101000100101011000011100110', 80),
)


@pytest.mark.parametrize('case,seed,taps,n_seq,sum_seq,head,sum_c', _LFSR_OCT,
                         ids=[c[0] for c in _LFSR_OCT])
def test_lfsr_against_octave(case, seed, taps, n_seq, sum_seq, head, sum_c):
    seq, c = _lfsr(seed, taps)
    seq = np.ravel(np.asarray(seq, dtype=int))
    c = np.asarray(c, dtype=int)

    assert seq.size == n_seq, (
        'seq has %d bits, COM Octave gives %d' % (seq.size, n_seq))
    assert ''.join(str(b) for b in seq[:len(head)]) == head, (
        'seq starts %s, COM Octave starts %s'
        % (''.join(str(b) for b in seq[:len(head)]), head))
    assert int(seq.sum()) == sum_seq, (
        'seq has %d ones, COM Octave has %d' % (int(seq.sum()), sum_seq))

    # c is the state history: 2^n - 1 rows of n bits, MATLAB c(k,:)
    assert c.shape == (2 ** len(seed) - 1, len(seed)), (
        'state history is %s, COM Octave gives (%d, %d)'
        % (c.shape, 2 ** len(seed) - 1, len(seed)))
    assert int(c.sum()) == sum_c, (
        'state history has %d ones, COM Octave has %d' % (int(c.sum()), sum_c))
    # ML 4197: c(1,:) = s, the seed before any shift
    assert list(c[0]) == list(seed)
    # ML 4211: seq = c(:,n)', the last column of the history
    assert list(seq) == list(c[:, len(seed) - 1])


def test_lfsr_does_not_mutate_the_caller_seed():
    """MATLAB passes s by value; the port takes a list and shifts it in place.

    _lfsr copies the seed on entry. Without that copy PRBS13Q's own seed
    literal would be rewritten on the first call and a second call in the same
    process would return a different sequence.
    """
    seed = [1, 0, 0, 1]
    before = list(seed)
    first, _ = _lfsr(seed, [4, 3])
    assert seed == before, 'the seed was shifted in place: %s' % seed
    second, _ = _lfsr(seed, [4, 3])
    assert list(np.ravel(first)) == list(np.ravel(second))


def test_lfsr_full_state_history_literal():
    """The whole register trajectory for the shortest case, written out.

    The parametrised tests above compare against a table, which is the right
    shape for seven cases but leaves no value visible in the assertion itself.
    This one spells the reference answer out: the 15 states COM Octave walks
    through for seed 1001 and taps [4 3], and the sequence LFSR reads off
    their last column.
    """
    seq, c = _lfsr([1, 0, 0, 1], [4, 3])
    assert ''.join(str(b) for b in np.ravel(seq)) == '100110101111000'
    assert int(np.asarray(c).sum()) == 32
    assert [''.join(str(b) for b in row) for row in np.asarray(c)] == [
        '1001', '1100', '0110', '1011', '0101', '1010', '1101', '1110',
        '1111', '0111', '0011', '0001', '1000', '0100', '0010']
