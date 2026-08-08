import numpy as np


def combines4p(s11in1, s12in1, s21in1, s22in1, s11in2, s12in2, s21in2, s22in2):
    """Cascade two 2-port S-parameter networks (MATLAB lines 5327-5370)."""
    def _sq(x): return np.asarray(x, dtype=complex).ravel()
    a11, a12 = _sq(s11in1), _sq(s12in1)
    a21, a22 = _sq(s21in1), _sq(s22in1)
    b11, b12 = _sq(s11in2), _sq(s12in2)
    b21, b22 = _sq(s21in2), _sq(s22in2)
    N = 1 - a22 * b11
    s11out = a11 + a12 * a21 * b11 / N
    s12out = a12 * b12 / N
    s21out = b21 * a21 / N
    s22out = b22 + b12 * b21 * a22 / N
    return s11out, s12out, s21out, s22out
