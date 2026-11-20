# ============================================================
# MATLAB->Python translation of fast_clip_topK
# MATLAB lines: 6249-6656 (com_ieee8023_4p18p0.m; new in 4p18p0)
# ============================================================
# ADC clipping level t with P(X > t) = P_qc, where X is the sampled pulse
# response times PAM symbols plus Gaussian noise. The K strongest cursors are
# enumerated exactly (L^K states); the rest, with the noise, are one Gaussian.
# t is found by bisection on the Gaussian-mixture tail, to a width of
# OP.BinSize. Called from get_PSDs when Clip Method = 'Fast_1' (4p18p0).
#
# Reproduced from the reference as it stands:
#   - The PAM levels come from `1:L/2`. For odd L the colon range stops below
#     L/2 (PAM3 gives the two levels [-1 1]/2) while there are still L
#     probabilities, so after the first cursor the state values and weights
#     differ in length and the first tail evaluation errors. COM Octave, L=3,
#     K=8: "product: nonconformant arguments (op1 is 1x6561, op2 is 1x256)".
#     Python raises the same error at the same point (upstream defect).
#   - ctle_signal_sigma uses sigma_X2*sum(h_top.^2), not the variance of the
#     enumerated states; the reference's own comment carries the latter, unused.
#   - DEBUG is hard-wired false; its prints and figures are not translated.
# ============================================================

import math

import numpy as np
from scipy.special import erfc


def _sort_abs_descend(h):
    """MATLAB [~,i] = sort(abs(h),'descend'): stable among ties, NaN first."""
    a = np.abs(h)
    nan = np.isnan(a)
    rest = np.flatnonzero(~nan)
    return np.concatenate([np.flatnonzero(nan), rest[np.argsort(-a[rest], kind='stable')]])


def fast_clip_topK(sampled_pulse_response, L, sigma_X2, sigma_noise, P_qc, OP, K):
    """Return (adc_clip, ctle_signal_sigma). MATLAB fast_clip_topK, 4p18p0."""
    h = np.asarray(sampled_pulse_response, dtype=float).ravel(order='F')
    N = h.size

    # m = L/2; odd_levels = 2*(1:m)-1 -- the colon range stops at floor(m)
    m = L / 2
    odd_levels = 2 * np.arange(1, math.floor(m) + 1, dtype=float) - 1
    a_levels = np.concatenate([-odd_levels[::-1], odd_levels]) / (L - 1)
    p_levels = np.ones(int(L)) / L

    idx_sorted = _sort_abs_descend(h)
    K_eff = min(K, N)
    idx_top = idx_sorted[:K_eff]
    idx_rest = idx_sorted[K_eff:]
    h_top = h[idx_top]
    h_rest = h[idx_rest]

    sigma_Srest2 = sigma_X2 * np.sum(h_rest ** 2)
    sigma_rest = np.sqrt(sigma_Srest2 + sigma_noise ** 2)

    # Exact PMF of S_top. sums(:) + hk*a_levels(:).' is states x levels, and
    # (:) flattens it column-major.
    sums = np.array([0.0])
    wts = np.array([1.0])
    for hk in h_top:
        sums = np.add.outer(sums, hk * a_levels).ravel(order='F')
        wts = np.multiply.outer(wts, p_levels).ravel(order='F')

    var_S_top = sigma_X2 * np.sum(h_top ** 2)
    ctle_signal_sigma = float(np.sqrt(var_S_top + sigma_rest ** 2))

    inv_sqrt2 = 1 / np.sqrt(2)
    inv_sigma_rest = 1 / sigma_rest
    scaled_sums = sums * inv_sigma_rest

    def tail_mix(t):
        if wts.size != scaled_sums.size:
            raise ValueError('fast_clip_topK: product: nonconformant arguments (op1 is 1x%d, '
                             'op2 is 1x%d), as in the reference: an odd PAM order (levels=%r) '
                             'gives fewer levels than symbol probabilities'
                             % (wts.size, scaled_sums.size, L))
        return np.sum(wts * (0.5 * erfc((t * inv_sigma_rest - scaled_sums) * inv_sqrt2)))

    Smax = np.sum(np.abs(h))
    t_lo = 0.0
    t_hi = Smax + 10 * sigma_rest

    tol = OP.BinSize
    while (t_hi - t_lo) > tol:
        t_mid = 0.5 * (t_lo + t_hi)
        if tail_mix(t_mid) > P_qc:
            t_lo = t_mid
        else:
            t_hi = t_mid

    adc_clip = float(t_hi)
    return adc_clip, ctle_signal_sigma
