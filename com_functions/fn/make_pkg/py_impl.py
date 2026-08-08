import numpy as np


def _synth_tline(f, Z_c, Z_0, gamma_coeff, tau, d):
    f = np.asarray(f, dtype=float).ravel()
    gc = np.asarray(gamma_coeff, dtype=float)
    f_GHz = f / 1e9
    gamma_1 = gc[1] * (1.0 + 1j)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_2 = gc[2] * (1.0 - 2j / np.pi * np.log(f_GHz)) + 2j * np.pi * tau
    gamma = gc[0] + gamma_1 * np.sqrt(f_GHz) + gamma_2 * f_GHz
    gamma[f_GHz == 0] = gc[0]
    rho_rl = 0.0 if d == 0 else (Z_c - 2.0 * Z_0) / (Z_c + 2.0 * Z_0)
    exp_gd = np.exp(-d * gamma)
    exp_gd2 = exp_gd ** 2
    denom = 1.0 - rho_rl ** 2 * exp_gd2
    s11 = rho_rl * (1.0 - exp_gd2) / denom
    s21 = (1.0 - rho_rl ** 2) * exp_gd / denom
    return s11, s21.copy(), s21.copy(), s11.copy()


def _combines4p(a11, a12, a21, a22, b11, b12, b21, b22):
    def sq(x): return np.asarray(x, dtype=complex).ravel()
    a11, a12, a21, a22 = sq(a11), sq(a12), sq(a21), sq(a22)
    b11, b12, b21, b22 = sq(b11), sq(b12), sq(b21), sq(b22)
    Nv = 1 - a22 * b11
    return (a11 + a12 * a21 * b11 / Nv, a12 * b12 / Nv,
            b21 * a21 / Nv, b22 + b12 * b21 * a22 / Nv)


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
