import numpy as np


def S_RN(f, G_DC, G_DC2, param):
    """Compute receiver noise PSD S_RN(f) = eta_0/2 * |H_CTF * H_R|^2."""
    f = np.asarray(f, dtype=float)
    p1 = float(np.asarray(param.CTLE_fp1).flat[0])
    z1 = float(np.asarray(param.CTLE_fz).flat[0])
    p2 = float(np.asarray(param.CTLE_fp2).flat[0])
    zlf = float(np.asarray(param.f_HP).flat[0])
    plf = zlf
    f_r = float(param.f_r)
    eta_0 = float(param.eta_0)
    H_CTF = (
        (10 ** (G_DC / 20) + 1j * f / z1)
        * (10 ** (G_DC2 / 20) + 1j * f / zlf)
        / ((1 + 1j * f / p1) * (1 + 1j * f / p2) * (1 + 1j * f / plf))
    )
    H_R = 1.0 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1], 1j * f / (f_r * param.fb))
    return eta_0 / 2 * np.abs(H_CTF * H_R) ** 2
