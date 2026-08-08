import numpy as np


def N_s(f, param, sigma_ns, OP):
    """Compute single-sided noise PSD Ns(f) for RX compliance injected noise."""
    f = np.asarray(f, dtype=float)
    f_b = float(param.fb)
    f_hp = float(param.f_hp)
    # MATLAB: inq = find(f <= f_b/2, 1, 'last')  [1-based]
    # Python: inq = last 0-based index where f <= f_b/2, +1 for exclusive-end slice
    mask = f <= f_b / 2
    inq = int(np.where(mask)[0][-1]) + 1 if np.any(mask) else 0
    RIT = str(OP.RIT_REF_PTR).lower()
    Ns_of_f = np.zeros(len(f))
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
