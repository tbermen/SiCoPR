import numpy as np
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)


from scipy.signal import fftconvolve
from types import SimpleNamespace


# ── Inlined helpers (from Group 3 implementations; no sibling imports) ────────


# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    """Build PDF from sampled-signal ISI via successive delta-set convolutions."""
    input_vector = np.asarray(input_vector, dtype=float).ravel()
    # MATLAB guards with `if max(abs(input_vector)) > BinSize ... else <delta>`.
    # On an empty vector max([]) is [] and `if []` is false, so MATLAB takes the
    # else branch; np.max raises on an empty array instead.  Reachable whenever
    # M exceeds the sample count, because then the late sub-phases are empty.
    # COM Octave, get_cm_noise(5, PR(1:3), 4, 1e-5, OP): results.CMn == 0.0256.
    if input_vector.size == 0 or _mmax(np.abs(input_vector)) <= BinSize:
        return _d_cpdf(BinSize, 0.0, 1.0)
    input_vector = input_vector[np.abs(input_vector) > BinSize]
    input_vector[np.abs(input_vector) < BinSize] = 0.0
    b = np.sign(input_vector)
    order = np.argsort(np.abs(input_vector), kind='stable')[::-1]
    input_vector = np.abs(input_vector)[order] * b[order]

    # MATLAB ones(1,L) accepts a non-integer-typed L; np.ones does not, and
    # param.levels arrives as a float (COM_FD_to_TD passes param.levels
    # straight through), so np.ones(4.0) raised TypeError on the real path.
    nlev = int(L)
    values = 2 * np.arange(nlev) / (L - 1) - 1   # Eq. 93A-39
    prob = np.ones(nlev) / L

    pdf = _d_cpdf(BinSize, 0.0, 1.0)
    empty_pdf = SimpleNamespace(**vars(pdf))
    for val in input_vector:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(val) * values, prob)
        pdf = _conv_fct(pdf, pdfn)
    return pdf


# ── Main function ─────────────────────────────────────────────────────────────

def get_cm_noise(M, PR, L, BER, OP=None):
    """Find best common-mode sub-phase and return CMn statistics.

    M: number of sub-phases.
    PR: waveform array, length = M * (len(PR) // M).
    L: number of PAM levels.
    BER: target BER for CMn calculation.
    OP: options struct; defaults OP.CM_norm_test=0 if not provided.
    Returns a SimpleNamespace with fields CMn, CMn_pdf, CMn_cdf, CMn_p2p.
    """
    if OP is None:
        OP = SimpleNamespace(CM_norm_test=0, DISPLAY_WINDOW=1)
    else:
        if not hasattr(OP, 'CM_norm_test'):
            OP.CM_norm_test = 0

    PR = np.asarray(PR, dtype=float).ravel()
    BinSize = 1e-5

    if M < 1:
        # MATLAB `for ki=1:M` runs zero times, so `results` is never created and
        # the caller's assignment errors out.  COM Octave, M=0:
        # "error: value on right hand side of assignment is undefined".
        raise ValueError('get_cm_noise: M must be at least 1 '
                         '(MATLAB leaves results undefined)')

    PR_fom_best = -np.inf
    results = SimpleNamespace()

    for ki in range(M):
        tps = PR[ki::M]                 # MATLAB: PR(ki:M:end)  (ki is 1-based → ki-1 in Python)
        if OP.CM_norm_test:
            PR_fom = float(np.linalg.norm(tps))
        else:
            testpdf = _get_pdf_from_sampled_signal(tps, L, BinSize * 10)
            cdf_test = np.cumsum(testpdf.y)
            hit = np.flatnonzero(cdf_test >= BER)
            # MATLAB find(...,1,'first') returns empty when the CDF never
            # reaches BER, so PRn_test is empty and `if PR_fom > PR_fom_best`
            # is false -- the best is left alone.  np.argmax on an all-false
            # mask returns 0, which invented -testpdf.x(1) as the answer.
            # COM Octave, BER=2: results.CMn == -Inf (Python gave 0.0701).
            PR_fom = None if hit.size == 0 else float(-testpdf.x[hit[0]])

        if PR_fom is not None and PR_fom > PR_fom_best:
            PR_fom_best = PR_fom

        if not OP.CM_norm_test:
            results.CMn = PR_fom_best
            results.CMn_pdf = testpdf
            results.CMn_cdf = cdf_test
        else:
            results.CMn = PR_fom_best
        results.CMn_p2p = float(_mmax(PR) - _mmin(PR))

    return results
