# ============================================================
# MATLAB→Python translation notes for scalePDF
# MATLAB lines: 11268–11275
# ============================================================
# 1-based vs 0-based: pdf_out.y(1)=pdf_out.y(2) → y[0]=y[1];
#                     pdf_out.y(end)=pdf_out.y(end-1) → y[-1]=y[-2].
# Range: (pdf_out.Min:-pdf_out.Min) → np.arange(Min, -Min+1).
# interp1 default (linear) returns NaN OUTSIDE the data range. np.interp
#   clamps to the end values instead, so it never produces a NaN and the
#   reference's two "NAN interp work around" lines become no-ops. That is only
#   harmless while at most ONE point falls outside at each end, which is what
#   the workaround patches. On a grid where max(x) < -min(x) -- a left-heavy
#   pdf -- many points fall outside and MATLAB returns NaN for all of them.
#   COM Octave, Min=-8, x=(-8:0)*0.05, delta_com=1: 19 NaNs, where clamping
#   gave 19 finite values. left=/right=nan restores it.
# Output shape: SimpleNamespace with same fields as input pdf.
# Known discrepancy from prior sicopr.py attempt: none found.
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

    # interp1(pdf.x*scale_factor, pdf.y, pdf_out.x) -- linear, NaN OUTSIDE the
    # data range. np.interp clamps unless told otherwise.
    pdf_out.y = np.interp(pdf_out.x,
                          np.asarray(pdf.x) * scale_factor,
                          np.asarray(pdf.y),
                          left=np.nan, right=np.nan)      # MATLAB line 11272

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
