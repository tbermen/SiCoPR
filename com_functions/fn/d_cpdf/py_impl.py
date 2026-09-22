# ============================================================
# MATLAB→Python translation notes for d_cpdf
# MATLAB lines: 5480–5530
# ============================================================
# all(values==0): early-return with scalar y=1, x=0 (special case).
# ~issorted(values): np.any(np.diff(values) < 0) → sort if not ascending.
# values=binsize*round(values/binsize): snap to grid.
# t=(values(1):binsize:values(end)): MATLAB range → np.arange(t_start, t_end+1)*binsize.
# pdf.Min = values(1)/binsize: 1-based first bin index (integer).
# Loop bin assignment:
#   k==1 (MATLAB) → bin=1 → Python k==0, bin_idx=0
#   k==length → last bin → Python k==N-1, bin_idx=len(t)-1
#   else: argmin(|t-v|) 0-based ✓
# pdf.y = pdf.y/sum(pdf.y): normalize to sum=1.
# support = find(pdf.y): 1-based non-zero indices → Python np.where > 0, 0-based.
# pdf.y = pdf.y(support(1):support(end)): trim leading/trailing zeros.
# pdf.Min += support(1)-1 (MATLAB 1-based shift) = support[0] (Python 0-based) ✓
# pdf.x = (pdf.Min:-pdf.Min)*binsize: symmetric axis → np.arange(Min,-Min+1)*binsize.
# ============================================================

import numpy as np

def _mround_arr(x):
    """MATLAB round() on an array: halves go away from zero, where np.round
    takes them to even.

    Only exact ties are corrected. Adding 0.5 and truncating would be wrong:
    0.49999999999999994 + 0.5 is exactly 1.0 in double precision, so that form
    rounds the largest double below a half up to 1 where MATLAB gives 0.
    """
    x = np.asarray(x, dtype=float)
    tie = np.abs(x - np.trunc(x)) == 0.5
    return np.where(tie, np.trunc(x) + np.copysign(1.0, x), np.round(x))

from types import SimpleNamespace


def d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if np.all(values == 0):
        pdf = SimpleNamespace()
        pdf.BinSize = binsize
        pdf.Min = 0
        pdf.y = np.array([1.0])
        pdf.x = np.array([0.0])
        return pdf

    if np.size(probs) < np.size(values):
        # MATLAB reads probs(k) for k = 1..length(values); a short probs is an
        # out-of-bound error, not a shorter answer.  zip() below would stop at
        # the shorter of the two and silently normalise whatever it collected.
        # COM Octave 4p16p0: d_cpdf(1,[-1 0 1],[0.5 0.5]) errors
        # "probs(3): out of bound 2 (dimensions are 1x2)".
        raise IndexError('d_cpdf: probs is shorter than values')

    # ~issorted(values): MATLAB requires every element <= the next, which is
    # false as soon as a NaN is present.  np.diff(values) < 0 is False across a
    # NaN, so that form calls [-1 NaN 1] sorted where MATLAB does not.
    if not np.all(values[:-1] <= values[1:]):     # ~issorted
        si = np.argsort(values, kind='stable')
        values = values[si]
        probs = probs[si]

    values = binsize * _mround_arr(values / binsize)   # snap to grid

    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize    # (values(1):binsize:values(end))

    pdf_min = t_start                               # values(1)/binsize (integer)
    pdf_y = np.zeros(len(t))

    for k, (v, prob) in enumerate(zip(values, probs)):
        if k == 0:
            bin_idx = 0                             # MATLAB k==1 → bin=1
        elif k == len(values) - 1:
            bin_idx = len(t) - 1                    # MATLAB k==length → last bin
        else:
            bin_idx = int(np.argmin(np.abs(t - v)))
        pdf_y[bin_idx] += prob

    pdf_y = pdf_y / np.sum(pdf_y)

    if np.any(pdf_y < 0):
        raise ValueError('PDF must be real and nonnegative')

    # find(pdf.y) selects *nonzero*, and NaN counts as nonzero.  `> 0` drops
    # NaN, so an all-zero or NaN-bearing probs vector (pdf.y = 0/0) left the
    # support empty and raised instead of answering.
    # COM Octave 4p16p0: d_cpdf(1,[-1 0 1],[0.5 NaN 0.5]) returns
    # Min=-1, x=[-1 0 1], y=[NaN NaN NaN]; likewise probs=[0 0 0].
    support = np.where(pdf_y != 0)[0]              # 0-based nonzero indices
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = pdf_min + int(support[0])             # MATLAB: pdf.Min+(support(1)-1)

    pdf = SimpleNamespace()
    pdf.BinSize = binsize
    pdf.Min = pdf_min
    pdf.y = pdf_y
    pdf.x = np.arange(pdf_min, -pdf_min + 1) * binsize   # (Min:-Min)*binsize
    return pdf


if __name__ == "__main__":
    out = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    print("Min:", out.Min, "x:", out.x, "y:", out.y)
