import numpy as np


def get_ILN(sdd21, faxis_f2, return_alpha=False):
    """Fit insertion loss normalisation curve and return ILN residual.

    Returns (ILN, efit) where:
      efit = weighted polynomial fit: a0 + a1*sqrt(f) + a2*f + a3*f^2
      ILN  = 20*log10(|sdd21|) - efit
    With return_alpha=True, (ILN, efit, alpha): 4p17p0 L7223 added alpha, the four
    fit coefficients, as a third output. No caller in the release reads it.
    """
    sdd21 = np.squeeze(np.asarray(sdd21, dtype=complex)).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    abs_s = np.abs(sdd21)
    # MATLAB is silent about log10(0) and about the 0*(-Inf) that follows it.
    with np.errstate(divide='ignore', invalid='ignore'):
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
        # warning('off','MATLAB:nearlySingularMatrix') immediately above it — i.e.
        # it deliberately takes the raw normal-equations inverse of a matrix it
        # knows is nearly singular, and keeps all four basis terms.
        #
        # This MUST NOT be replaced by np.linalg.lstsq. faxis is in Hz, so the f^2
        # column reaches ~4.5e21 and cond(fmbg'fmbg) overflows to inf; lstsq then
        # truncates small singular values and solves with an effective rank of 2 of
        # 4, silently discarding half the fit basis. That produced fitted-IL errors
        # of 3.6-14.4 dB against MATLAB and a correspondingly wrong FOM_ILD.
        A = fmbg.T @ fmbg
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            # MATLAB's inv() does not fail on an exactly singular matrix: it
            # warns and returns Inf in every entry, and that Inf (or the NaN
            # that Inf*0 makes of it) flows out through efit and ILN.
            # COM Octave, get_ILN on a single frequency point:
            #   warning: inverse: matrix singular to machine precision, rcond = 0
            #   alpha = [Inf Inf Inf Inf]   efit = Inf   ILN = -Inf
            # and on an all-zero sdd21:
            #   alpha = [NaN NaN NaN NaN]   efit = NaN   ILN = NaN
            # Falling back to lstsq answered both with finite numbers the
            # reference never produces: efit 1.0662767 where MATLAB says Inf,
            # and efit 0 where MATLAB says NaN.
            Ainv = np.full_like(A, np.inf)
        # MATLAB associates left to right: ((fmbg'*fmbg)^-1 * fmbg') * LGw.
        alpha = (Ainv @ fmbg.T) @ LGw
        efit = (
            alpha[0]
            + alpha[1] * np.sqrt(faxis_f2)
            + alpha[2] * faxis_f2
            + alpha[3] * faxis_f2 ** 2
        )
        ILN = db_s - efit
    if return_alpha:
        return ILN, efit, alpha
    return ILN, efit
