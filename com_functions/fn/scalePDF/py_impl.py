# ============================================================
# MATLAB→Python translation notes for scalePDF
# MATLAB lines: 11268–11275
# ============================================================
# 1-based vs 0-based: pdf_out.y(1)=pdf_out.y(2) → y[0]=y[1];
#                     pdf_out.y(end)=pdf_out.y(end-1) → y[-1]=y[-2].
# Range: (pdf_out.Min:-pdf_out.Min) → np.arange(Min, -Min+1).
# interp1 default (linear, NaN outside): np.interp clips to boundary
#   instead of NaN; the neighbour-copy workaround is still applied for
#   fidelity, and is harmless when np.interp already returned a value.
# Output shape: SimpleNamespace with same fields as input pdf.
# Known discrepancy from prior com.py attempt: none found.
# ============================================================
import copy
import numpy as np


def scalePDF(pdf, scale_factor):
    """Scale the x-axis of a PDF struct and re-interpolate the y-values.

    pdf.Min is floored after scaling; the x-grid is recomputed and y is
    linearly interpolated from the original (scaled) data then renormalised.
    """
    pdf_out = copy.copy(pdf)                               # shallow copy of struct

    pdf_out.Min = int(np.floor(pdf.Min * scale_factor))   # MATLAB line 11270

    # (pdf_out.Min:-pdf_out.Min) → integer range inclusive on both ends
    idx = np.arange(pdf_out.Min, -pdf_out.Min + 1)
    pdf_out.x = idx * pdf_out.BinSize                     # MATLAB line 11271

    # interp1(pdf.x*scale_factor, pdf.y, pdf_out.x) — linear, clamp at edges
    pdf_out.y = np.interp(pdf_out.x,
                          np.asarray(pdf.x) * scale_factor,
                          np.asarray(pdf.y))              # MATLAB line 11272

    # NaN workaround (MATLAB lines 11273-11274): copy neighbours at edges
    pdf_out.y[0]  = pdf_out.y[1]
    pdf_out.y[-1] = pdf_out.y[-2]

    pdf_out.y = pdf_out.y / np.sum(pdf_out.y)             # MATLAB line 11275
    return pdf_out


if __name__ == "__main__":
    from types import SimpleNamespace
    import numpy as np
    p = SimpleNamespace()
    p.Min = -4
    p.BinSize = 0.5
    p.x = np.arange(-4, 5) * 0.5          # [-2, -1.5, ..., 2]
    p.y = np.exp(-p.x**2 / 0.5)
    p.y /= p.y.sum()
    out = scalePDF(p, 2.0)
    print("Min:", out.Min, "len(x):", len(out.x), "sum(y):", out.y.sum())
