"""Verification tests for pam() — Grey-coded PAM4 NRZ-pair mapping.

# ============================================================
# MATLAB GROUND TRUTH
# Grey-code mapping (from source):
#   [-1,-1] → -1       (PAM4 level 0)
#   [-1, 1] → -1/3     (PAM4 level 1)
#   [ 1, 1] →  1/3     (PAM4 level 2)
#   [ 1,-1] →  1       (PAM4 level 3)
#
# pam([-1,-1, -1,1, 1,1, 1,-1]) → [-1, -1/3, 1/3, 1]
# pam([-1,-1]) → [-1]
# pam([-1, 1]) → [-1/3]
# pam([ 1, 1]) → [ 1/3]
# pam([ 1,-1]) → [  1 ]
# Odd-length input: last element ignored (floor(N/2) pairs).
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.pam.py_impl import pam


def test_all_four_symbols():
    bits = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    result = pam(bits)
    np.testing.assert_allclose(result, [-1.0, -1.0/3, 1.0/3, 1.0], rtol=1e-12)


def test_level_minus1():
    assert pam(np.array([-1.0, -1.0]))[0] == pytest.approx(-1.0)


def test_level_minus_third():
    assert pam(np.array([-1.0, 1.0]))[0] == pytest.approx(-1.0/3)


def test_level_plus_third():
    assert pam(np.array([1.0, 1.0]))[0] == pytest.approx(1.0/3)


def test_level_plus1():
    assert pam(np.array([1.0, -1.0]))[0] == pytest.approx(1.0)


def test_output_length():
    """Output length is floor(input_length / 2)."""
    bits = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    assert len(pam(bits)) == 3


def test_odd_length_truncates():
    """Odd-length input: last element ignored."""
    bits = np.array([-1.0, -1.0, 1.0])   # 1 complete pair
    assert len(pam(bits)) == 1
    assert pam(bits)[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Values below are from the executed reference, not a reading of it: each was
# produced by running pam() from octave/com_ieee8023_4p16p0_octave_compat.m
# under COM Octave via tools/octave_oracle.py. The three unassigned-slot cases
# corrected what I had assumed -- in particular pam([1 1 0 0]) is length ONE.
# ---------------------------------------------------------------------------

def test_oracle_nominal():
    """COM Octave: pam([-1 -1 -1 1 1 1 1 -1]) -> [-1 -1/3 1/3 1]"""
    out = pam(np.array([-1., -1, -1, 1, 1, 1, 1, -1]))
    np.testing.assert_allclose(out, [-1.0, -1.0 / 3, 1.0 / 3, 1.0])
    assert out.shape == (4,)


def test_oracle_unmatched_pair_backfills_zero():
    """COM Octave: pam([0 0 1 1]) -> [0 1/3].

    MATLAB takes no if/elseif branch for [0 0], leaving dataout(1) unassigned;
    assigning dataout(2) then grows the array and back-fills index 1 with 0.
    """
    np.testing.assert_allclose(pam(np.array([0., 0, 1, 1])), [0.0, 1.0 / 3])


def test_oracle_trailing_unmatched_pair_truncates():
    """COM Octave: pam([1 1 0 0]) -> [1/3], LENGTH 1.

    The mirror image of the previous case: MATLAB grows dataout only as far as
    the highest assigned index, so an unmatched FINAL pair shortens the result
    rather than back-filling a zero.
    """
    out = pam(np.array([1., 1, 0, 0]))
    np.testing.assert_allclose(out, [1.0 / 3])
    assert out.shape == (1,), 'trailing unmatched pair must not be back-filled'


def test_oracle_odd_length_drops_last():
    """COM Octave: pam([-1 -1 1]) -> [-1]; the loop runs to floor(N/2)*2."""
    np.testing.assert_allclose(pam(np.array([-1., -1, 1])), [-1.0])


@pytest.mark.parametrize('bad', [[1.0], []])
def test_oracle_too_short_raises(bad):
    """COM Octave errors: 'value on right hand side of assignment is undefined'.

    MATLAB never assigns dataout, so the function has no output argument.
    Returning an empty array here would hand the caller a silent wrong answer.
    """
    with pytest.raises(ValueError):
        pam(np.array(bad, dtype=float))
