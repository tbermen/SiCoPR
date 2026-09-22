# ============================================================
# MATLAB→Python translation notes for find_eye_width
# MATLAB lines: 5723–5785
# ============================================================
# eye_contour is (N×2): column 0 = top eye, column 1 = bottom eye.
# MATLAB 1-based indexing: eye_contour(half_UI:-1:1, 1) → Python eye_contour[half_UI-1::-1, 0].
# half_UI appears in both the left slice (reversed) and right slice (forward).
# vref_intersect: linear interpolation between the two samples bracketing the vref crossing.
#   x_in is 1-based within the passed slice.
# ============================================================

import numpy as np


def _vref_intersect(eye_col_1d, x_in_1based, vref):
    # MATLAB reads eye_contour(x_in-1,1), so x_in<=1 asks for subscript 0 and
    # errors. eye_col_1d[x-2] turns that into a NEGATIVE index and quietly
    # reads the LAST row instead, answering off the far end of the eye.
    # COM Octave: vref_intersect([0.1;0.3;0.7;0.9], 1, 0.5) ->
    #   "eye_contour(0,_): subscripts must be either integers 1 to (2^63)-1
    #    or logicals"                 (this copy returned 0.5)
    # Same guard as the canonical vref_intersect; this copy takes the eye
    # column already sliced out, so it indexes one dimension rather than two.
    if x_in_1based != int(x_in_1based) or x_in_1based < 2:
        raise IndexError('eye_contour(%s,_): subscripts must be either '
                         'integers 1 to (2^63)-1 or logicals'
                         % (x_in_1based - 1,))
    x = int(x_in_1based)
    m1 = eye_col_1d[x - 1] - eye_col_1d[x - 2]
    b1 = eye_col_1d[x - 1] - m1 * x
    return (vref - b1) / m1


def find_eye_width(eye_contour, half_UI, samples_per_UI, vref):
    eye_contour = np.asarray(eye_contour)

    # Left eye (top)
    left_top = eye_contour[half_UI - 1::-1, 0]
    idx = np.where(left_top < vref)[0]
    if len(idx) == 0:
        L1 = half_UI
    elif idx[0] == 0:
        L1 = 0
    else:
        vc = int(idx[0]) + 1  # 1-based
        x_in = half_UI - vc + 2
        INT = _vref_intersect(eye_contour[:half_UI, 0], x_in, vref)
        L1 = half_UI - INT

    # Left eye (bottom)
    left_bot = eye_contour[half_UI - 1::-1, 1]
    idx = np.where(left_bot > vref)[0]
    if len(idx) == 0:
        L0 = half_UI
    elif idx[0] == 0:
        L0 = 0
    else:
        vc = int(idx[0]) + 1
        x_in = half_UI - vc + 2
        INT = _vref_intersect(eye_contour[:half_UI, 1], x_in, vref)
        L0 = half_UI - INT

    # Right eye (top)
    right_top = eye_contour[half_UI - 1:, 0]
    idx = np.where(right_top < vref)[0]
    if len(idx) == 0:
        R1 = samples_per_UI - half_UI
    elif idx[0] == 0:
        R1 = 0
    else:
        vc = int(idx[0]) + 1
        INT = _vref_intersect(eye_contour[half_UI - 1:, 0], vc, vref) + half_UI - 1
        R1 = INT - half_UI

    # Right eye (bottom)
    right_bot = eye_contour[half_UI - 1:, 1]
    idx = np.where(right_bot > vref)[0]
    if len(idx) == 0:
        R0 = samples_per_UI - half_UI
    elif idx[0] == 0:
        R0 = 0
    else:
        vc = int(idx[0]) + 1
        INT = _vref_intersect(eye_contour[half_UI - 1:, 1], vc, vref) + half_UI - 1
        R0 = INT - half_UI

    Left_EW = min(L1, L0)
    Right_EW = min(R1, R0)
    return Left_EW, Right_EW
