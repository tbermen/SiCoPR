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
from types import SimpleNamespace


def Init_PDF_Fast(EmptyPDF, values, probs):
    pdf = SimpleNamespace(**vars(EmptyPDF))         # pdf = EmptyPDF (copy)
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)

    rvd = np.round(values / pdf.BinSize).astype(int)   # rounded_values_div_binsize

    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])

    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]                              # 0-based bin placement (MATLAB: rvd-Min+1)

    pdf.y[bp[0]] = probs[0]                        # first value: direct assign
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]

    return pdf


if __name__ == "__main__":
    empty = SimpleNamespace(BinSize=0.1, Min=0, y=np.array([0.0]), x=np.array([0.0]))
    out = Init_PDF_Fast(empty, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    print("x:", out.x, "y:", out.y, "Min:", out.Min)
