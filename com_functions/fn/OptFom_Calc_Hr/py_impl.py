# ============================================================
# MATLAB→Python translation notes for OptFom_Calc_Hr
# MATLAB lines: 2874–2880
# ============================================================
# H_r = product of Bessel_Thomson_Filter, Butterworth_Filter and
#   Raised_Cosine_Filter (element-wise multiply).
#
# The three filters are IMPORTED from their sibling py_impl.py files rather
# than inlined. The private copies that used to live here were the pre-oracle
# forms and had already been proved wrong in their own directories:
#   - `_tukey_window` was an element-wise np.where, but MATLAB's Tukey_Window
#     CONCATENATES three counted pieces, so the answer is grouped by category
#     and only lines up with f when f ascends.
#     COM Octave (RC_Start=20e9, RC_end=40e9, OP.Raised_Cosine on):
#       f=[25e9 0 35e9 15e9]         -> [1 1 0.85355339059327373 0.14644660940672616]
#       (the inlined np.where gave   [0.85355339059327373 1 0.14644660940672616 1])
#       f=[20e9 20e9 40e9 40e9 0 90e9] -> [1 1 1 0 0 0]
#       (the inlined np.where gave   [1 1 0 0 1 0])
#   - `if not use_BW` raised on any numpy array of other than one element and
#     took the filter branch for [1, 0]; MATLAB's `if` is true only for a
#     non-empty value whose elements are ALL non-zero.
#   - `np.ones(len(f))` is not `ones(1,length(f))`: length() is the LONGEST
#     dimension, so a 2x3 f gives three ones, not six.
# A private copy is exactly how those fixes failed to arrive here, which is why
# this now imports. param fields needed: fb, fb_BW_cutoff (BW),
# BTorder/fb_BT_cutoff (BT), RC_Start/RC_end (RC).
# ============================================================

from com_functions.fn.Bessel_Thomson_Filter.py_impl import Bessel_Thomson_Filter
from com_functions.fn.Butterworth_Filter.py_impl import Butterworth_Filter
from com_functions.fn.Raised_Cosine_Filter.py_impl import Raised_Cosine_Filter


def OptFom_Calc_Hr(f, param, OP):
    H_bt = Bessel_Thomson_Filter(param, f, OP.Bessel_Thomson)
    H_bw = Butterworth_Filter(param, f, OP.Butterworth)
    H_RCos = Raised_Cosine_Filter(param, f, OP.Raised_Cosine)
    return H_bw * H_bt * H_RCos


if __name__ == "__main__":
    import numpy as np
    from types import SimpleNamespace
    param = SimpleNamespace(fb=25e9, fb_BW_cutoff=1.0,
                            BTorder=2, fb_BT_cutoff=1.0,
                            RC_Start=5e9, RC_end=20e9)
    OP = SimpleNamespace(Bessel_Thomson=False, Butterworth=False, Raised_Cosine=False)
    H = OptFom_Calc_Hr(np.array([0.0, 10e9, 25e9]), param, OP)
    print("All disabled:", H)
