# ============================================================
# MATLAB→Python translation notes for Tukey_Window
# MATLAB lines: 4677–4696
# ============================================================
# ~exist('fr','var') && ~exist('fb','var'): optional positional args
#   → Python default fr=None, fb=None; both None → use param fields
# Three counted pieces, concatenated (NOT three element-wise regions):
#   f < fr  → 1 (passband)
#   fr≤f≤fb → 0.5*cos(2π*(f-fb)/fperiod - π) + 0.5 (raised-cosine rolloff)
#   f > fb  → 0 (stopband)
# H_tw=H_tw(1:length(f)): the pieces run long when fr>fb makes the categories
#   overlap, and short when a value (a NaN) lands in none of them
# Boundary check: at f=fr → H=1; at f=fb → H=0 ✓
# ============================================================

import numpy as np


def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def Tukey_Window(f, param, fr=None, fb=None):
    f = np.asarray(f, dtype=float)
    if fr is None and fb is None:
        fb = param.fb
        fr = param.f_r * param.fb
    fperiod = 2 * (fb - fr)
    # MATLAB CONCATENATES three counted pieces — ones(1,n<fr), the raised
    # cosine of the in-band values, zeros(1,n>fb) — so the answer is grouped by
    # category and only lines up with f when f ascends.  Element-wise np.where
    # silently returned a different vector for any other order.  COM Octave,
    # fr=1e9 fb=3e9:
    #   f=[1e9 1e9 3e9 3e9 0 9e9] -> [1 1 1 0 0 0]   (element-wise: [1 1 0 0 1 0])
    #   f=[2.5e9 0 3.5e9 1.5e9]   -> [1 0.14644660940672616 0.85355339059327373 0]
    flat = np.atleast_1d(f).ravel(order='F')   # MATLAB linear-index order
    n_lo = int(np.count_nonzero(flat < fr))
    n_hi = int(np.count_nonzero(flat > fb))
    band = flat[(flat >= fr) & (flat <= fb)]
    mid = 0.5 * np.cos(2 * np.pi * (band - fb) / fperiod - np.pi) + 0.5
    # Only the middle piece keeps the orientation of f, so for a column or a
    # matrix MATLAB's horizontal concatenation fails unless that piece has at
    # most one element or is the only non-empty one.  COM Octave, column f:
    # "horizontal dimensions mismatch (1x2 vs 5x1)".
    if f.ndim > 1 and f.shape[0] != 1 and mid.size > 1 and (n_lo or n_hi):
        raise ValueError('Tukey_Window: horizontal dimensions mismatch '
                         '(1x%d vs %dx1)' % (n_lo or n_hi, mid.size))
    H_tw = np.concatenate([np.ones(n_lo), mid, np.zeros(n_hi)])
    n = _length(f)
    # The pieces cover every element of f only while each one lands in exactly
    # one category.  A NaN lands in none, so H_tw comes up short and MATLAB's
    # H_tw(1:length(f)) is an out-of-bound read.  COM Octave, f=[0 1.5e9 NaN
    # 3.5e9]: "H_tw(4): out of bound 3".  np.where answered 0 for the NaN.
    if H_tw.size < n:
        raise IndexError('Tukey_Window: H_tw(%d): out of bound %d' % (n, H_tw.size))
    return H_tw[:n]
