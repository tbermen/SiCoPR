import numpy as np


def N_s(f, param, sigma_ns, OP):
    """Compute single-sided noise PSD Ns(f) for RX compliance injected noise."""
    f = np.asarray(f, dtype=float)
    n_out = 0 if f.size == 0 else (max(f.shape) if f.ndim else 1)   # MATLAB length()
    # MATLAB's Ns_of_f is always the row zeros(1,length(f)) and is written
    # through linear indexing, so a COLUMN f is answered, not refused.  The
    # (inq,1)-into-(inq,) assignment here raised "could not broadcast input
    # array from shape (3,1) into shape (3,)".  COM Octave, column f =
    # [0 25 50 75 100]e9 with clause_179, returns the 1x5 row
    # [0 2.3770725964419546e-15 2.6513502037237188e-15 0 0].
    f = f.ravel(order='F')          # MATLAB linear-index order
    f_b = float(param.fb)
    f_hp = float(param.f_hp)
    # MATLAB: inq = find(f <= f_b/2, 1, 'last')  [1-based]
    # Python: inq = last 0-based index where f <= f_b/2, +1 for exclusive-end slice
    mask = f <= f_b / 2
    inq = int(np.where(mask)[0][-1]) + 1 if np.any(mask) else 0
    RIT = str(OP.RIT_REF_PTR).lower()
    # max(): indexing past the end grows the array in MATLAB, which a matrix f
    # can reach (length() counts only the longest dimension while find() walks
    # every element).  COM Octave, f 2x3 with clause_178, returns 1x4.
    Ns_of_f = np.zeros(max(n_out, inq))
    if RIT == 'clause_178':
        Ns_of_f[:inq] = 2 * sigma_ns ** 2 / f_b
    elif RIT in ('clause_179', 'annex_176d'):
        if f_hp <= 0:
            raise ValueError('Parameter f_hp must be greater than 0')
        beta = 1 - (2 * f_hp / f_b) * np.arctan(f_b / (2 * f_hp))
        Ns_of_f[:inq] = (
            (2 * sigma_ns ** 2) / (beta * f_b)
            * (f[:inq] / f_hp) ** 2
            / (1 + (f[:inq] / f_hp) ** 2)
        )
    else:
        raise ValueError(f'unsupported RIT Reference Pointer "{RIT}"')
    return Ns_of_f
