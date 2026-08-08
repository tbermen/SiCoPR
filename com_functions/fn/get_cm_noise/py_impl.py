import numpy as np
from types import SimpleNamespace


# ── Inlined helpers (from Group 3 implementations; no sibling imports) ────────

def _d_cpdf(binsize, values, probs):
    """Create a discrete PDF struct from values and probabilities."""
    values = np.asarray(values, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    if np.all(values == 0):
        p = SimpleNamespace(BinSize=binsize, Min=0,
                            y=np.array([1.0]), x=np.array([0.0]))
        return p
    if np.any(np.diff(values) < 0):
        si = np.argsort(values)
        values, probs = values[si], probs[si]
    values = binsize * np.round(values / binsize)
    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize
    pdf_y = np.zeros(len(t))
    for k, (v, pr) in enumerate(zip(values, probs)):
        if k == 0:
            bi = 0
        elif k == len(values) - 1:
            bi = len(t) - 1
        else:
            bi = int(np.argmin(np.abs(t - v)))
        pdf_y[bi] += pr
    pdf_y /= np.sum(pdf_y)
    support = np.where(pdf_y > 0)[0]
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = t_start + int(support[0])
    return SimpleNamespace(
        BinSize=binsize, Min=pdf_min, y=pdf_y,
        x=np.arange(pdf_min, -pdf_min + 1) * binsize
    )


def _Init_PDF_Fast(EmptyPDF, values, probs):
    """Fast PDF initialisation from an EmptyPDF template."""
    pdf = SimpleNamespace(**vars(EmptyPDF))
    values = np.asarray(values, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    rvd = np.round(values / pdf.BinSize).astype(int)
    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])
    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


def _conv_fct(p1, p2):
    """Convolve two PDF structs."""
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')
    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = np.convolve(np.asarray(p1.y, dtype=float), np.asarray(p2.y, dtype=float))
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    """Build PDF from sampled-signal ISI via successive delta-set convolutions."""
    input_vector = np.asarray(input_vector, dtype=float).ravel()
    if np.max(np.abs(input_vector)) <= BinSize:
        return _d_cpdf(BinSize, 0.0, 1.0)
    input_vector = input_vector[np.abs(input_vector) > BinSize]
    input_vector[np.abs(input_vector) < BinSize] = 0.0
    b = np.sign(input_vector)
    order = np.argsort(np.abs(input_vector))[::-1]
    input_vector = np.abs(input_vector)[order] * b[order]

    values = 2 * np.arange(L) / (L - 1) - 1   # Eq. 93A-39
    prob = np.ones(L) / L

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

    PR_fom_best = -np.inf
    results = SimpleNamespace()

    for ki in range(M):
        tps = PR[ki::M]                 # MATLAB: PR(ki:M:end)  (ki is 1-based → ki-1 in Python)
        if OP.CM_norm_test:
            PR_fom = float(np.linalg.norm(tps))
        else:
            testpdf = _get_pdf_from_sampled_signal(tps, L, BinSize * 10)
            cdf_test = np.cumsum(testpdf.y)
            first_idx = int(np.argmax(cdf_test >= BER))
            PRn_test = float(-testpdf.x[first_idx])
            PR_fom = PRn_test

        if PR_fom > PR_fom_best:
            PR_fom_best = PR_fom

        if not OP.CM_norm_test:
            results.CMn = PR_fom_best
            results.CMn_pdf = testpdf
            results.CMn_cdf = cdf_test
        else:
            results.CMn = PR_fom_best
        results.CMn_p2p = float(np.max(PR) - np.min(PR))

    return results
