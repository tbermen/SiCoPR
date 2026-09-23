# ============================================================
# MATLAB→Python translation notes for scaleCDF
# MATLAB lines: 11257–11267
# ============================================================
# Calls scalePDF internally; inlined as _scale_pdf (protocol: no sibling imports).
# P = cumsum(pdf.y): CDF from PDF y array.
# ider0 = find(P>=DER0,1,'first'): 1-based in MATLAB → np.argmax(P>=DER0) 0-based.
#   Both index into the same array position → no offset needed ✓
# anias = pdf.x(ider0)/A_s: voltage at DER0 crossing / signal amplitude.
# scale_factor = 1/10^(-delta_com/20) = 10^(delta_com/20).
# new_value: computed for fidelity but not returned (dead code in MATLAB).
# pdf_out = scalePDF(pdf, scale_factor): inline _scale_pdf.
# cdf_out = cumsum(pdf_out.y): CDF of scaled PDF.
# ============================================================

import copy
import numpy as np
from types import SimpleNamespace


def _scale_pdf(pdf, scale_factor):
    """Inline of scalePDF: scale x-axis and re-interpolate y."""
    pdf_out = copy.copy(pdf)
    pdf_out.Min = int(np.floor(pdf.Min * scale_factor))
    idx = np.arange(pdf_out.Min, -pdf_out.Min + 1)
    pdf_out.x = idx * pdf_out.BinSize
    xs = np.asarray(pdf.x, dtype=float) * scale_factor
    pdf_out.y = np.interp(pdf_out.x, xs, np.asarray(pdf.y, dtype=float))
    # interp1's default is linear with NaN OUTSIDE the data range; np.interp
    # clamps to the end values instead.  The two-sample "NAN interp work
    # around" below hides that only while the source grid is symmetric.  When
    # max(pdf.x) < -min(pdf.x) the new grid runs past the data on the right by
    # more than one bin and MATLAB returns NaN everywhere (sum is then NaN).
    # COM Octave: Min=-8, x=(-8:0)*0.05, delta_com=1 -> pdf_out.y is 19 NaNs;
    # np.interp alone produced 19 finite values.
    pdf_out.y[(pdf_out.x < xs[0]) | (pdf_out.x > xs[-1])] = np.nan
    pdf_out.y[0] = pdf_out.y[1]
    pdf_out.y[-1] = pdf_out.y[-2]
    pdf_out.y = pdf_out.y / np.sum(pdf_out.y)
    return pdf_out


def scaleCDF(pdf, delta_com, DER0, A_s):
    pdf_out = copy.copy(pdf)
    P = np.cumsum(np.asarray(pdf.y, dtype=float))
    # find(...,1,'first') is EMPTY when the CDF never reaches DER0, and the
    # next line is then `-1/[]`, which the reference refuses.  COM Octave,
    # DER0=5 on a normalised pdf: "operator /: nonconformant arguments
    # (op1 is 1x1, op2 is 1x0)".  np.argmax on an all-False mask returns 0,
    # so the port answered with the wrong bin and no sign of trouble.
    if not np.any(P >= DER0):
        raise ValueError('scaleCDF: cumsum(pdf.y) never reaches DER0=%r; '
                         'MATLAB errors on the empty find() (-1/[])' % (DER0,))
    ider0 = int(np.argmax(P >= DER0))           # 0-based; equiv to MATLAB 1-based find
    anias = pdf.x[ider0] / A_s
    new_db = 20 * np.log10(-1.0 / anias) - delta_com
    new_value = -1.0 / 10 ** (new_db / 20)     # computed for fidelity; not returned
    scale_factor = 1.0 / 10 ** (-delta_com / 20)
    pdf_out = _scale_pdf(pdf, scale_factor)
    cdf_out = np.cumsum(pdf_out.y)
    return pdf_out, cdf_out, scale_factor


if __name__ == "__main__":
    pdf = SimpleNamespace(
        BinSize=0.1, Min=-2,
        x=np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
        y=np.array([0.20, 0.30, 0.30, 0.15, 0.05]),
    )
    pdf_out, cdf_out, sf = scaleCDF(pdf, 0.0, 0.4, 1.0)
    print("scale_factor:", sf, " cdf_out[-1]:", cdf_out[-1])
