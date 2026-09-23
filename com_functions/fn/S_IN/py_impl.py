import numpy as np


def _N_s(f, param, sigma_ns, OP):
    """Inlined helper: single-sided noise PSD Ns(f).

    Kept in step with com_functions/fn/N_s/py_impl.py; the two bodies must
    stay identical.  This copy had not picked up the column-input and
    matrix-input corrections made there, so S_IN raised on a call MATLAB
    answers -- see the COM Octave block in test_verify.py.
    """
    f = np.asarray(f, dtype=float)
    n_out = 0 if f.size == 0 else (max(f.shape) if f.ndim else 1)   # MATLAB length()
    f = f.ravel(order='F')          # MATLAB linear-index order
    f_b = float(param.fb)
    f_hp = float(param.f_hp)
    mask = f <= f_b / 2
    inq = int(np.where(mask)[0][-1]) + 1 if np.any(mask) else 0
    RIT = str(OP.RIT_REF_PTR).lower()
    # max(): indexing past the end grows the array in MATLAB, which a matrix f
    # can reach (length() counts only the longest dimension while find() walks
    # every element).
    Ns = np.zeros(max(n_out, inq))
    if RIT == 'clause_178':
        Ns[:inq] = 2 * sigma_ns ** 2 / f_b
    elif RIT in ('clause_179', 'annex_176d'):
        if f_hp <= 0:
            raise ValueError('Parameter f_hp must be greater than 0')
        beta = 1 - (2 * f_hp / f_b) * np.arctan(f_b / (2 * f_hp))
        Ns[:inq] = (
            (2 * sigma_ns ** 2) / (beta * f_b)
            * (f[:inq] / f_hp) ** 2 / (1 + (f[:inq] / f_hp) ** 2)
        )
    else:
        raise ValueError(f'unsupported RIT Reference Pointer "{RIT}"')
    return Ns


def S_IN(f, noise_path, G_DC, G_DC2, param, OP):
    """Compute injected noise PSD S_IN(f) = Ns/2 * |Hn21 * H_CTF * H_R|^2."""
    f = np.asarray(f, dtype=float)
    noise_path = np.asarray(noise_path, dtype=complex)
    p1 = float(np.asarray(param.CTLE_fp1).flat[0])
    z1 = float(np.asarray(param.CTLE_fz).flat[0])
    p2 = float(np.asarray(param.CTLE_fp2).flat[0])
    zlf = float(np.asarray(param.f_HP).flat[0])
    plf = zlf
    f_r = float(param.f_r)
    # eps(0) in MATLAB = smallest positive float; keeps sigma_ns nonzero
    sigma_ns = float(param.sigma_ns) + np.nextafter(0.0, 1.0)
    H_CTF = (
        (10 ** (G_DC / 20) + 1j * f / z1)
        * (10 ** (G_DC2 / 20) + 1j * f / zlf)
        / ((1 + 1j * f / p1) * (1 + 1j * f / p2) * (1 + 1j * f / plf))
    )
    H_R = 1.0 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1], 1j * f / (f_r * param.fb))
    Ns_of_f = _N_s(f, param, sigma_ns, OP)
    return Ns_of_f / 2 * np.abs(noise_path * H_CTF * H_R) ** 2
