# MATLAB lines 4843-4857, with SL (4368-4401), R_series2 (4354-4360), r_parrelell2 (9390-9395)

import copy
import numpy as np

from com_functions.fn.make_full_pkg.py_impl import make_full_pkg


def _combines4p(a11, a12, a21, a22, b11, b12, b21, b22):
    N = 1.0 - a22 * b11
    return (
        a11 + a12 * a21 * b11 / N,
        a12 * b12 / N,
        b21 * a21 / N,
        b22 + b12 * b21 * a22 / N,
    )


def _R_series2(zref, nfreq, R):
    """MATLAB R_series2: series resistance 2-port S-params."""
    s11 = np.full(nfreq, R / (R + 2.0 * zref), dtype=complex)
    s21 = np.full(nfreq, 2.0 * zref / (R + 2.0 * zref), dtype=complex)
    return s11, s21, s21, s11  # s11, s12, s21, s22


def _r_parrelell2(zref, nfreq, rpad):
    """MATLAB r_parrelell2: parallel resistance 2-port S-params."""
    s11 = np.full(nfreq, -zref / (rpad * (zref / rpad + 2.0)), dtype=complex)
    s21 = np.full(nfreq, 2.0 / (zref / rpad + 2.0), dtype=complex)
    return s11, s21, s21, s11  # s11, s12, s21, s22


def _SL(S, R, zref):
    """MATLAB SL: cascade a series/parallel R element before S."""
    SLD = copy.deepcopy(S)
    if R == zref:
        return SLD

    nfreq = S.Parameters.shape[0]
    s11c = S.Parameters[:, 0, 0]
    s12c = S.Parameters[:, 0, 1]
    s21c = S.Parameters[:, 1, 0]
    s22c = S.Parameters[:, 1, 1]

    if R > zref:
        spr11, spr12, spr21, spr22 = _R_series2(zref, nfreq, R - zref)
    else:
        rpad = -R * zref / (R - zref)
        spr11, spr12, spr21, spr22 = _r_parrelell2(zref, nfreq, rpad)

    o11, o12, o21, o22 = _combines4p(spr11, spr12, spr21, spr22,
                                      s11c, s12c, s21c, s22c)
    SLD.Parameters[:, 0, 0] = o11
    SLD.Parameters[:, 0, 1] = o12
    SLD.Parameters[:, 1, 0] = o21
    SLD.Parameters[:, 1, 1] = o22
    return SLD


def add_pkg_with_die(S, mode, param, OP):
    """Add TX package with die resistance to channel S-params.

    MATLAB lines 4843-4857.

    S: SimpleNamespace with .Parameters (nfreq,2,2), .Frequencies, .Impedance, .NumPorts
    Returns (Smx, TX_RL).
    """
    # Step 1: Build TX package S-params
    s11in, s12in, s21in, s22in = make_full_pkg(
        'TX', S.Frequencies, param, 'THRU', mode)

    # Step 2: Cascade package → channel
    s11c = S.Parameters[:, 0, 0]
    s12c = S.Parameters[:, 0, 1]
    s21c = S.Parameters[:, 1, 0]
    s22c = S.Parameters[:, 1, 1]

    o11, o12, o21, o22 = _combines4p(s11in, s12in, s21in, s22in,
                                      s11c, s12c, s21c, s22c)

    Smx = copy.deepcopy(S)
    Smx.Parameters[:, 0, 0] = o11
    Smx.Parameters[:, 0, 1] = o12
    Smx.Parameters[:, 1, 0] = o21
    Smx.Parameters[:, 1, 1] = o22

    # Step 3: Apply die-pad series resistance (SL)
    R_diepad = float(np.asarray(param.R_diepad).ravel()[0])
    Z0 = float(param.Z0)
    if mode.lower() in ('dd', 'cd'):
        Rd, zref = R_diepad * 2.0, Z0 * 2.0
    else:
        Rd, zref = R_diepad / 2.0, Z0 * 2.0

    Smx = _SL(Smx, Rd, zref)

    # Step 4: TX_RL = S22 of combined (MATLAB Smx.Parameters(2,2,:))
    TX_RL = Smx.Parameters[:, 1, 1].copy()

    # Step 5: Restore S11 (TX-side reflection = original channel, not including pkg)
    Smx.Parameters[:, 0, 0] = S.Parameters[:, 0, 0]

    return Smx, TX_RL
