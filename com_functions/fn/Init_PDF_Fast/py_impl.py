# ============================================================
# MATLAB→Python translation notes for Init_PDF_Fast
# MATLAB lines: 2199–2265
# ============================================================
# EmptyPDF: template struct; at minimum has BinSize field.
# rounded_values_div_binsize = round(values/BinSize): integer bin indices.
# pdf.x = BinSize*rvd(1) : BinSize : BinSize*rvd(end)
#   → np.arange(rvd[0], rvd[-1]+1) * BinSize  (integer-bin range ×BinSize)
# pdf.Min = rvd(1): first (lowest) bin index.
# bin_placement = rvd - pdf.Min + 1 (1-based in MATLAB)
#   → rvd - rvd[0] (0-based in Python)
# pdf.y(bin_placement(1)) = probs(1): direct assignment (zero-init, so = +=)
# Loop k=2..N: accumulate (+= for duplicate-bin values).
# No normalisation — caller ensures sum(probs)==1 or normalises afterward.
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


def Init_PDF_Fast(EmptyPDF, values, probs):
    pdf = SimpleNamespace(**vars(EmptyPDF))         # pdf = EmptyPDF (copy)
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)

    rvd = _mround_arr(values / pdf.BinSize).astype(int)   # rounded_values_div_binsize

    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])

    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]                              # 0-based bin placement (MATLAB: rvd-Min+1)

    if np.any(bp < 0) or np.any(bp >= len(pdf.y)):
        # pdf.x only spans rvd(1)..rvd(end), so any value that rounds outside
        # that span (i.e. `values` is not ascending) makes bin_placement fall
        # off the array and MATLAB stops.  A negative index is legal in numpy,
        # so Python wrapped round and added the probability to the wrong bin.
        # COM Octave 4p16p0: Init_PDF_Fast(E,[0 -0.2 0.3],[0.2 0.3 0.5]) with
        # BinSize=0.1 errors "pdf(-1): subscripts must be either integers
        # 1 to (2^63)-1 or logicals"; Python answered y=[0.2 0 0.3 0.5].
        raise IndexError('Init_PDF_Fast: values must be ascending')

    pdf.y[bp[0]] = probs[0]                        # first value: direct assign
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]

    return pdf


if __name__ == "__main__":
    empty = SimpleNamespace(BinSize=0.1, Min=0, y=np.array([0.0]), x=np.array([0.0]))
    out = Init_PDF_Fast(empty, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    print("x:", out.x, "y:", out.y, "Min:", out.Min)
