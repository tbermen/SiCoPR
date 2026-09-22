# ============================================================
# MATLAB→Python translation notes for synth_tline
# MATLAB lines: 11292–11316
# ============================================================
# 1-based vs 0-based: gamma_coeff(1),(2),(3) → [0],[1],[2].
# log(f_GHz) at f=0 is -inf → gamma_2*f_GHz = 0*(-inf) = NaN.
#   Handled by gamma[f_GHz==0] = gamma_coeff[0] override (line 11300).
#   Use np.errstate to suppress the divide/invalid warnings.
# sqrt(f_GHz): real for f≥0; no complex issues.
# Output shape: four 1-D arrays (s11,s12,s21,s22) of length N.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def synth_tline(f, Z_c, Z_0, gamma_coeff, tau, d):
    """Synthesize transmission-line 2-port S-parameters (IEEE 802.3 93A model).

    Parameters
    ----------
    f           : frequency array (Hz)
    Z_c         : characteristic impedance of the line (Ω)
    Z_0         : reference impedance (Ω)
    gamma_coeff : 3-element array [c0, c1, c2] — loss model coefficients
    tau         : group delay (s/m? — per-length phase coefficient)
    d           : length (same units as tau denominator)

    Returns s11, s12, s21, s22 — each a 1-D complex array of length N.
    """
    f = np.asarray(f, dtype=float).ravel()
    gamma_coeff = np.asarray(gamma_coeff, dtype=float)

    f_GHz = f / 1e9                                  # MATLAB line 11293
    # MATLAB's sqrt() and log() of a negative real return a complex value;
    # numpy's on a float array return NaN.  Casting makes the port answer as
    # the reference does for f<0, and is bit-neutral for f>=0.
    # COM Octave: synth_tline(f=[-53e9 -1e9 -1 0 1 1e9 53e9], Z_c=100, Z_0=50,
    #   gamma_coeff=[2e-4 3.4e-3 1e-4], tau=1e-11, d=0.15) returns finite
    #   complex s21 at f<0 (1.006069830981237-0.0057570668335386373i at -53GHz),
    #   where the float path gave NaN.
    f_GHz_c = f_GHz.astype(complex)

    # Eq 93A-10
    gamma_1 = gamma_coeff[1] * (1.0 + 1j)           # MATLAB line 11295

    # Eq 93A-11 — suppress log(0) warning; NaN overridden at line 11300
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_2 = (gamma_coeff[2] * (1.0 - 2j / np.pi * np.log(f_GHz_c))
                   + 2j * np.pi * tau)               # MATLAB line 11297

        # Eq 93A-9
        gamma = (gamma_coeff[0]
                 + gamma_1 * np.sqrt(f_GHz_c)
                 + gamma_2 * f_GHz)                  # MATLAB line 11299
    gamma[f_GHz == 0] = gamma_coeff[0]               # MATLAB line 11300 (DC fix)

    # Eq 93A-12
    if d == 0:                                       # MATLAB line 11303
        rho_rl = 0.0
    else:
        # numpy division, not Python's: at Z_c == -2*Z_0 MATLAB divides by zero
        # and carries Inf through to an all-NaN result rather than raising.
        # COM Octave: synth_tline(f, Z_c=-100, Z_0=50, ...) -> s11, s21 all NaN.
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_rl = np.float64(Z_c - 2.0 * Z_0) / np.float64(Z_c + 2.0 * Z_0)
                                                     # MATLAB line 11308

    exp_gd = np.exp(-d * gamma)                      # MATLAB line 11311
    exp_gd2 = exp_gd ** 2
    denom = 1.0 - rho_rl ** 2 * exp_gd2

    # Eqs 93A-13, 93A-14
    s11 = rho_rl * (1.0 - exp_gd2) / denom          # MATLAB line 11313
    s21 = (1.0 - rho_rl ** 2) * exp_gd / denom      # MATLAB line 11314
    s12 = s21.copy()                                 # MATLAB line 11315
    s22 = s11.copy()                                 # MATLAB line 11316

    return s11, s12, s21, s22


if __name__ == "__main__":
    import numpy as np
    f = np.array([0.0, 1e9, 10e9])
    gc = np.array([0.0, 0.1, 0.02])
    s11, s12, s21, s22 = synth_tline(f, 100.0, 50.0, gc, 0.0, 0.1)
    print("s11:", s11)
    print("s21:", s21)
