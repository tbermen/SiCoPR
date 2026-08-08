import numpy as np


def get_ILN(sdd21, faxis_f2):
    """Fit insertion loss normalisation curve and return ILN residual.

    Returns (ILN, efit) where:
      efit = weighted polynomial fit: a0 + a1*sqrt(f) + a2*f + a3*f^2
      ILN  = 20*log10(|sdd21|) - efit
    """
    sdd21 = np.squeeze(np.asarray(sdd21, dtype=complex)).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    abs_s = np.abs(sdd21)
    db_s = 20.0 * np.log10(abs_s)

    # Weighted basis matrix (n x 4); columns are abs_s * [1, sqrt(f), f, f^2]
    fmbg = np.column_stack([
        abs_s,
        np.sqrt(faxis_f2) * abs_s,
        faxis_f2 * abs_s,
        faxis_f2 ** 2 * abs_s,
    ])
    LGw = abs_s * db_s  # weighted log response (RHS)

    # Least-squares solve (MATLAB used normal equations with warning suppressed)
    alpha, _, _, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)
    efit = (
        alpha[0]
        + alpha[1] * np.sqrt(faxis_f2)
        + alpha[2] * faxis_f2
        + alpha[3] * faxis_f2 ** 2
    )
    ILN = db_s - efit
    return ILN, efit
