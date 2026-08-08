# ============================================================
# MATLAB→Python translation notes for combine_pdf_same_voltage_axis
# MATLAB lines: 5286–5326
# ============================================================
# Unlike comb_fct, this function works with explicit voltage x arrays
#   rather than integer bin indices (Min).
# min1/min2: first element of x arrays (leftmost voltage).
# shift_amount = round(|min1-min2| / BinSize) → integer number of bins.
# Left-alignment: prepend x/y from the further-left pdf to the other:
#   MATLAB pdf1.x(1:shift_amount) → Python x1[:shift_amount]
#   (1-based [1:n] → 0-based [:n])
# Right-alignment: zero-pad the shorter of the two after left-alignment.
# Output has same BinSize, combined voltage axis, and summed probabilities.
# No renormalization — caller is responsible (per MATLAB comment).
# When min1==min2: shift_amount=0; else branch is a no-op → direct addition.
# ============================================================

import numpy as np
from types import SimpleNamespace


def combine_pdf_same_voltage_axis(pdf1, pdf2):
    if pdf1.BinSize != pdf2.BinSize:
        raise ValueError('bin size must be equal')

    x1 = np.asarray(pdf1.x, dtype=float).copy()
    y1 = np.asarray(pdf1.y, dtype=float).copy()
    x2 = np.asarray(pdf2.x, dtype=float).copy()
    y2 = np.asarray(pdf2.y, dtype=float).copy()

    min1 = x1[0]   # pdf1.x(1) in MATLAB = x1[0] in Python
    min2 = x2[0]
    shift_amount = int(round(abs(min1 - min2) / pdf1.BinSize))

    if min1 < min2:
        # pdf1 extends further left; prepend first shift_amount pts of x1 to x2/y2
        x2 = np.concatenate([x1[:shift_amount], x2])
        y2 = np.concatenate([np.zeros(shift_amount), y2])
    else:
        # pdf2 extends further left (or equal); prepend first shift_amount pts of x2 to x1/y1
        x1 = np.concatenate([x2[:shift_amount], x1])
        y1 = np.concatenate([np.zeros(shift_amount), y1])

    L1 = len(x1)
    L2 = len(x2)
    Ldiff = abs(L1 - L2)

    if L1 > L2:
        out_x = x1
        y2 = np.concatenate([y2, np.zeros(Ldiff)])
    else:
        out_x = x2
        y1 = np.concatenate([y1, np.zeros(Ldiff)])

    out_pdf = SimpleNamespace()
    out_pdf.x = out_x
    out_pdf.y = y1 + y2
    out_pdf.BinSize = pdf1.BinSize
    return out_pdf


if __name__ == "__main__":
    pdf1 = SimpleNamespace(x=np.array([-0.2,-0.1,0.0,0.1,0.2]),
                           y=np.array([1.0,2,3,2,1]), BinSize=0.1)
    pdf2 = SimpleNamespace(x=np.array([-0.1,0.0,0.1]),
                           y=np.array([1.0,2,1]), BinSize=0.1)
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    print("x:", out.x)
    print("y:", out.y)   # expect [1,3,5,3,1]
