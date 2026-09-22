import numpy as np
from com_functions.fn.combines4p.py_impl import combines4p as _combines4p
from com_functions.fn.synth_tline.py_impl import synth_tline as _synth_tline


def make_pkg(f, pkg_len, cpad, cball, pkg_z, param, *varargin):
    """Build package S-parameters: pad capacitor + transmission line + ball capacitor (MATLAB lines 8359-8405).

    Optional varargin[0] = lcomp (compensating inductor), varargin[1] = cbump (bump capacitor).
    Returns (s11, s12, s21, s22) as 1-D complex arrays.
    """
    f = np.asarray(f, dtype=float).ravel()
    f = np.where(f < np.finfo(float).eps, np.finfo(float).eps, f)  # f(f<eps)=eps

    tau = float(param.pkg_tau)
    gc = param.pkg_gamma0_a1_a2
    Lenscale = float(pkg_len)
    zref = float(param.Z0)

    # Pad capacitor (Eq. 93A-8)
    denom_pad = 2 + 1j * 2 * np.pi * f * float(cpad) * zref
    s11pad = -1j * 2 * np.pi * f * float(cpad) * zref / denom_pad
    s21pad = 2.0 / denom_pad
    s12pad = s21pad.copy()
    s22pad = s11pad.copy()

    # Optional compensating inductor
    if len(varargin) > 0:
        lcomp = float(varargin[0])
        if lcomp > 0:
            denom_c = 2 + 1j * 2 * np.pi * f * lcomp / zref
            s11comp = (1j * 2 * np.pi * f * lcomp / zref) / denom_c
            s21comp = 2.0 / denom_c
            s11pad, s12pad, s21pad, s22pad = _combines4p(
                s11pad, s12pad, s21pad, s22pad, s11comp, s21comp, s21comp, s11comp)

    # Optional bump capacitor
    if len(varargin) > 1:
        cbump = float(varargin[1])
        if cbump > 0:
            denom_b = 2 + 1j * 2 * np.pi * f * cbump * zref
            s11bump = -1j * 2 * np.pi * f * cbump * zref / denom_b
            s21bump = 2.0 / denom_b
            s11pad, s12pad, s21pad, s22pad = _combines4p(
                s11pad, s12pad, s21pad, s22pad, s11bump, s21bump, s21bump, s11bump)

    # Transmission line segment
    S11, S12, S21, S22 = _synth_tline(f, float(pkg_z), zref, gc, tau, Lenscale)
    s11out1, s12out1, s21out1, s22out1 = _combines4p(
        s11pad, s12pad, s21pad, s22pad, S11, S21, S21, S11)

    # Ball capacitor (Eq. 93A-8)
    denom_ball = 2 + 1j * 2 * np.pi * f * float(cball) * zref
    s11ball = -1j * 2 * np.pi * f * float(cball) * zref / denom_ball
    s21ball = 2.0 / denom_ball

    s11out, s12out, s21out, s22out = _combines4p(
        s11out1, s12out1, s21out1, s22out1, s11ball, s21ball, s21ball, s11ball)

    return s11out, s12out, s21out, s22out
