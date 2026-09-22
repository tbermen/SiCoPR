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


# --------------------------------------------------------------------------
# Against COM Octave's own add_pkg_with_die and make_full_pkg.
#
# This function had 16.4% line coverage and no assertions at all. It builds
# the TX package and die network and cascades it onto the channel, which is
# the block a historical defect already hid in (the die network truncated to
# one of three LC sections).
#
# Expected values are COM Octave's, via tools/octave_oracle.py.
#
# One finding is held rather than fixed: with a MATRIX C_diepad/L_comp, the
# kind the shipped workbooks carry, the two implementations disagree. See
# test_matrix_die_parameters_match_com_octave below.
# --------------------------------------------------------------------------

import numpy as np  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from com_functions.fn.make_full_pkg.py_impl import make_full_pkg  # noqa: E402

_NF, _FMAX = 16, 40e9
_OCT_MFP = {  # make_full_pkg('TX', f, param, 'THRU', 'dd'), vector die params
    's11': -0.0013015936032590159 - 0.0072846021326057487j,
    's21': 0.99165140263455431 - 0.055114342899675192j,
    's22': -0.0013050944038014815 - 0.0072840238264049824j,
}
_OCT_APD = {  # add_pkg_with_die(S, 'dd', param, OP)
    'S11': 0.039921069137130866 - 0.0025116207811725346j,
    'S12': 0.92055846804503738 - 0.16839159087775382j,
    'S22': 0.002358652248722018 + 0.0041161212154938696j,
    'TX_RL_first': 0.002358652248722018 + 0.0041161212154938696j,
    'TX_RL_last': 0.0015331496527624978 + 0.0077574571621451737j,
}


def _channel():
    f = np.linspace(1e8, _FMAX, _NF)
    mag = 10 ** (-(0.5 * np.sqrt(f / 1e9)) / 20)
    s21 = mag * np.exp(-1j * 2 * np.pi * f * 2e-10)
    s11 = 0.04 * np.exp(-1j * 2 * np.pi * f * 1e-10)
    P = np.zeros((_NF, 2, 2), dtype=complex)      # SiCoPR order: (nfreq, 2, 2)
    P[:, 0, 0] = s11
    P[:, 0, 1] = s21
    P[:, 1, 0] = s21
    P[:, 1, 1] = s11
    return f, P


def _pkg_param(matrix_die=False):
    cd = (np.array([[4e-14, 9e-14, 1.1e-13], [4e-14, 9e-14, 1.1e-13]])
          if matrix_die else np.array([4e-14, 9e-14]))
    lc = (np.array([[1.3e-10, 1.5e-10, 1.4e-10], [1.3e-10, 1.5e-10, 1.4e-10]])
          if matrix_die else np.array([1.3e-10, 1.5e-10]))
    pk = SimpleNamespace(pkg_gamma0_a1_a2=np.array([0, 1.734e-3, 1.455e-4]),
                         pkg_tau=6.141e-3)
    return SimpleNamespace(
        Z0=50, R_diepad=np.array([46.25, 46.25]), C_diepad=cd, L_comp=lc,
        C_bump=np.array([3e-14, 3e-14]), C_v=np.array([0.0, 0.0]),
        C_pkg_board=np.array([0.0, 0.0]),
        Pkg_len_TX=np.array([12, 1.8, 0, 0]), Pkg_len_RX=np.array([12, 1.8, 0, 0]),
        Pkg_len_NEXT=np.array([12, 1.8, 0, 0]),
        Pkg_len_FEXT=np.array([12, 1.8, 0, 0]),
        pkg_Z_c=np.array([[87.5, 95, 100, 100], [87.5, 95, 100, 100]]),
        PKG_NAME=['A', 'A'], PKG={'A': pk}, z_p_next_cases=1)


def _close(got, want, tol=1e-12):
    return abs(complex(got) - want) <= tol * max(1.0, abs(want))


def test_make_full_pkg_matches_com_octave():
    f, _P = _channel()
    s11, _s12, s21, s22 = make_full_pkg('TX', f, _pkg_param(), 'THRU', 'dd')
    for name, got in (('s11', np.asarray(s11).ravel()[0]),
                      ('s21', np.asarray(s21).ravel()[0]),
                      ('s22', np.asarray(s22).ravel()[0])):
        assert _close(got, _OCT_MFP[name]), (
            '%s[0] is %r, COM Octave gives %r' % (name, got, _OCT_MFP[name]))


def test_add_pkg_with_die_matches_com_octave():
    f, P = _channel()
    S = SimpleNamespace(Parameters=P, Frequencies=f, Impedance=100, NumPorts=2)
    Smx, TX_RL = add_pkg_with_die(S, 'dd', _pkg_param(), SimpleNamespace())
    par = np.asarray(Smx.Parameters)
    got = {'S11': par[0, 0, 0], 'S12': par[0, 0, 1], 'S22': par[0, 1, 1],
           'TX_RL_first': np.asarray(TX_RL).ravel()[0],
           'TX_RL_last': np.asarray(TX_RL).ravel()[-1]}
    for name, want in _OCT_APD.items():
        assert _close(got[name], want), (
            '%s is %r, COM Octave gives %r' % (name, got[name], want))


def test_die_elements_actually_change_the_result():
    """Positive control: with include_die=0 the die is zeroed, and that answer
    must differ from the one above, or the package test is only exercising the
    transmission line. (Both engines agree exactly in that zeroed case.)"""
    f, _P = _channel()
    with_die = np.asarray(make_full_pkg('TX', f, _pkg_param(), 'THRU', 'dd')[0]).ravel()[0]
    no_die = np.asarray(
        make_full_pkg('TX', f, _pkg_param(), 'THRU', 'dd', 0)[0]).ravel()[0]
    assert abs(with_die - no_die) > 1e-6, (
        'zeroing the die changed s11 by only %.2e; the die model is not being '
        'applied' % abs(with_die - no_die))


@pytest.mark.xfail(strict=True, reason=(
    'HELD FOR REVIEW 2026-09-22: with a matrix C_diepad/L_comp -- the shape '
    'the shipped workbooks carry -- SiCoPR and COM Octave disagree. SiCoPR '
    'returns the answer COM Octave gives for the equivalent VECTOR die '
    'parameters, so it appears to take the first Tx/Rx pair where the '
    'reference consumes the matrix differently. With vector parameters the '
    'two agree to 1e-16, and with the die zeroed they agree exactly, so the '
    'transmission line is not involved. Whether the engine ever reaches '
    'make_full_pkg with a matrix, or selects the package-case row upstream, '
    'decides whether this matters. Weighing against it: the 208-case MATLAB '
    'correlation is bit-exact on COM with workbooks whose C_diepad IS a '
    'matrix, and that is this same path reached with real parameters and '
    'agreeing -- not an untested path excused by a corpus. So the likelier '
    'reading is that the synthetic matrix here is inconsistent with the '
    'Pkg_len/pkg_Z_c shapes the matrix branch expects (num_blocks becomes '
    'mele + len(Cd_Tx) - 1). Resolve by capturing a real param set.'))
def test_matrix_die_parameters_match_com_octave():
    f, _P = _channel()
    s11 = np.asarray(
        make_full_pkg('TX', f, _pkg_param(matrix_die=True), 'THRU', 'dd')[0]).ravel()[0]
    # COM Octave, same matrix parameters
    assert _close(s11, -0.0013811167357231284 - 0.008590681999406666j)
