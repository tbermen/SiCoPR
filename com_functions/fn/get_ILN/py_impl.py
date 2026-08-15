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

    # MATLAB: alpha = ((fmbg'*fmbg)^-1)*fmbg'*LGw, with
    # warning('off','MATLAB:nearlySingularMatrix') immediately above it — i.e. it
    # deliberately takes the raw normal-equations inverse of a matrix it knows is
    # nearly singular, and keeps all four basis terms.
    #
    # This MUST NOT be replaced by np.linalg.lstsq. faxis is in Hz, so the f^2
    # column reaches ~4.5e21 and cond(fmbg'fmbg) overflows to inf; lstsq then
    # truncates small singular values and solves with an effective rank of 2 of 4,
    # silently discarding half the fit basis. That produced fitted-IL errors of
    # 3.6-14.4 dB against MATLAB and a correspondingly wrong FOM_ILD.
    A = fmbg.T @ fmbg
    rhs = fmbg.T @ LGw
    try:
        alpha = np.linalg.inv(A) @ rhs
    except np.linalg.LinAlgError:          # exactly singular — MATLAB would warn and
        alpha, _, _, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)   # return Inf/NaN
    if not np.all(np.isfinite(alpha)):
        alpha, _, _, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)
    efit = (
        alpha[0]
        + alpha[1] * np.sqrt(faxis_f2)
        + alpha[2] * faxis_f2
        + alpha[3] * faxis_f2 ** 2
    )
    ILN = db_s - efit
    return ILN, efit
