# ============================================================
# MATLAB→Python translation notes for conv_fct_MeanNotZero
# MATLAB lines: 5389–5404
# ============================================================
# Identical logic to conv_fct — different function name only.
# Name indicates use for PDFs whose mean may be non-zero (the Min bin
# index can be non-symmetric), whereas conv_fct targets symmetric PDFs.
# Both use the same convolution formula: p.Min = round(p1.Min+p2.Min)
# and x axis (p.Min*BinSize : BinSize : pMax*BinSize).
# ============================================================

import numpy as np
from types import SimpleNamespace



# ML: p.y = conv2(p1.y, p2.y) -- a DIRECT convolution, and the port must be one.
#
# This dispatched to scipy's fftconvolve once both operands reached 128 bins,
# for speed, on the claim that the two agree "to ~1e-15 relative". Relative to
# the PEAK they do. Per element they do not: an FFT buries every value below
# about eps times the peak in round-off, negative "probabilities" included,
# and the far tail of the noise CDF is exactly where DER_DFE and DER_MLSE are
# read. COM Octave on woXtalk_T1_R19: CDF agreement fell from 4e-13 relative
# at 1e-3 to 1.3e-4 at 1e-12, then Octave 5.2e-221 against SiCoPR 2.1e-16,
# and DER_DFE came out 4.5 percent wrong while every other stage matched.
# A speed-up that changes a result is not one this port may keep.


# a kernel of at least _SPARSE_MIN bins with at most one nonzero bin in
# _SPARSE_RATIO is convolved in Octave's order over its nonzero bins
_SPARSE_MIN = 64
_SPARSE_RATIO = 8


def _conv1d(a, b):
    """Convolve two 1-D PDFs directly, as MATLAB conv2 does."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # conv2 with an empty operand returns empty; np.convolve raises instead.
    # COM Octave: p1.y=[1 2 3], p2.y=[] -> p.y is 0x0, p.x is 1x0, p.Min=-1.
    if a.size == 0 or b.size == 0:
        return np.zeros(0)
    # Convolve only the nonzero SPAN of each operand and put the exact zeros
    # back around it. Every product dropped has a zero factor, so what changes
    # is only the order in which BLAS sums the rest -- ulp level, never the tail
    # (the round-off that ruled out the FFT scales with each element, not with
    # the peak). The big win is get_PSDs' ADC-clip PDF: a signal PDF against a
    # Gaussian laid on the same ~12k-bin axis, ~55% of it underflowed to zero,
    # ~1200 times per case -- 0.3-0.5 s each in full, ~8 ms on the span.
    # Accepted 2026-09-24 under the owner's equivalence-class rule: on all 28
    # checkpoint cases every reported output other than the DER family is
    # bit-identical to full-array convolution, the rest moves by ulps, and
    # agreement with COM Octave is no worse on any field.
    if b.size >= _SPARSE_MIN:
        nzb = np.flatnonzero(b)
        if nzb.size * _SPARSE_RATIO <= b.size:
            # A sparse KERNEL -- the 4-level delta sets get_pdf_from_sampled_
            # signal convolves in ~10^5-10^6 times per case, four nonzero bins
            # across hundreds or thousands. conv2 (liboctave oct-convn) builds
            # the result one kernel element at a time, c += b(k)*a in
            # ascending k, so summing over the nonzero bins only is Octave's
            # own order: on a realistic ISI build this matched COM Octave's
            # conv_fct on 6049 of 6049 elements, np.convolve on 5602.
            out = np.zeros(a.size + b.size - 1)
            n = a.size
            for k in nzb:
                out[k:k + n] += b[k] * a
            return out
    if a[0] != 0 and a[-1] != 0 and b[0] != 0 and b[-1] != 0:
        # nonzero at both ends: the span is the whole array, and this is the
        # call the span path below would make -- without scanning ~10^4 bins
        # on each of ~200k calls to find that out
        return np.convolve(a, b)
    ia = np.flatnonzero(a)
    ib = np.flatnonzero(b)
    out = np.zeros(a.size + b.size - 1)
    if ia.size == 0 or ib.size == 0:
        return out
    a0, a1 = ia[0], ia[-1] + 1
    b0, b1 = ib[0], ib[-1] + 1
    out[a0 + b0:a1 + b1 - 1] = np.convolve(a[a0:a1], b[b0:b1])
    return out


def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is half-to-even.

    COM Octave: p1.Min=-0.5, p2.Min=0 -> p.Min=-1  (Python round() gives 0)
    """
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    return int(round(x))


def _colon_x(pmin, pmax, binsize):
    """MATLAB `pmin*binsize : binsize : pmax*binsize`.

    The MATLAB comment right above this line claims the colon is "equivalent
    to (p.Min:p.Min+length(p.y)-1)*p.BinSize". It is not: the colon
    accumulates from the first element as a+k*d and pins the last element to
    the stated limit, while (a:b)*d forms each element as one product. They
    differ by 1 ulp on most bins.

    COM Octave: p.Min=-5, BinSize=1e-4, 9 bins ->
      [-0.00050000000000000001, -0.00040000000000000002, -0.00030000000000000003,
       -0.00019999999999999998, -9.9999999999999991e-05, 0,
        0.00010000000000000005,  0.00019999999999999998,  0.00030000000000000003]
    where np.arange(-5,4)*1e-4 differs on several bins. Swept over 7920
    (Min, length, BinSize) combinations against Octave: the product form got
    249387 of 1013684 elements wrong (24.6%, all by 1 ulp); this form got none.

    The last element is pinned to the limit only when accumulation overshoots
    it, which is what the colon does -- pinning unconditionally is wrong, e.g.
    COM Octave: p.Min=-3, BinSize=0.1, 2 bins ->
      [-0.30000000000000004, -0.20000000000000004]   (not ..., -0.2)

    Known gap: in 38 of those 7920 combinations Octave's colon yields one
    element FEWER than length(p.y), so p.x is shorter than p.y. That is not
    reproduced here (nor by the old form); it needs Octave's fuzzy element
    count, and every rule tried for it broke far more cases than it fixed.
    """
    n = pmax - pmin + 1
    if n <= 0:
        return np.zeros(0)
    a = pmin * binsize
    b = pmax * binsize
    x = a + np.arange(n) * binsize
    if (binsize > 0 and x[-1] > b) or (binsize < 0 and x[-1] < b):
        x[-1] = b
    return x


def conv_fct_MeanNotZero(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')

    p = SimpleNamespace(**vars(p1))
    p.Min = _mround(p1.Min + p2.Min)
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = _colon_x(p.Min, pMax, p.BinSize)
    return p


if __name__ == "__main__":
    p1 = SimpleNamespace(Min=2, BinSize=0.1, y=np.array([0.6, 0.4]))
    p2 = SimpleNamespace(Min=1, BinSize=0.1, y=np.array([0.7, 0.3]))
    out = conv_fct_MeanNotZero(p1, p2)
    print("Min:", out.Min, "x:", out.x, "y:", out.y)
