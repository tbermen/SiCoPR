# ============================================================
# MATLAB→Python translation notes for TD_CTLE
# MATLAB lines: 4593–4609
# ============================================================
# Bilinear-transform CTLE applied to an impulse response.
# poly([a,b]) → np.poly([a,b]): same convention — roots to monic polynomial
#   highest-degree first, np.poly returns float array ✓
# filter(B,A,x) → scipy.signal.lfilter(B,A,x): direct-form IIR filter ✓
# All intermediate values (p1d, p2d, zd) are real scalars since
#   p1_ctle/bilinear_fs is real.
# kd formula: (bilinear_fs−z_ctle)/((bilinear_fs−p1_ctle)*(bilinear_fs−p2_ctle)) * f_p1/f_z
#   normalises DC gain with respect to the ratio f_p1/f_z.
# Returns 4 values: (impulse_response, p1_ctle, p2_ctle, z_ctle)
# ============================================================

import numpy as np
from scipy.signal import lfilter


def TD_CTLE(ir_in, fb, f_z, f_p1, f_p2, kacdc_dB, oversampling):
    # No dtype=float: MATLAB's filter() carries a complex input through, and
    # the cast silently DISCARDED the imaginary part (a ComplexWarning only).
    # COM Octave, ir = [1 0 0 0 0 0 0 1j]: the imaginary delta re-appears as
    # 0.66006149825361637j in the last sample.
    # atleast_1d: MATLAB filters a scalar (1x1) and returns a scalar;
    # lfilter raised "selected axis is out of range" on a 0-d array.
    ir_in = np.atleast_1d(np.asarray(ir_in))
    p1_ctle = -2 * np.pi * f_p1
    p2_ctle = -2 * np.pi * f_p2
    z_ctle = -2 * np.pi * f_z * 10 ** (kacdc_dB / 20)
    k_ctle = -p2_ctle
    bilinear_fs = 2 * fb * oversampling
    p2d = (1 + p2_ctle / bilinear_fs) / (1 - p2_ctle / bilinear_fs)
    p1d = (1 + p1_ctle / bilinear_fs) / (1 - p1_ctle / bilinear_fs)
    zd = (1 + z_ctle / bilinear_fs) / (1 - z_ctle / bilinear_fs)
    kd = ((bilinear_fs - z_ctle)
          / ((bilinear_fs - p1_ctle) * (bilinear_fs - p2_ctle))
          * f_p1 / f_z)
    B_filt = k_ctle * kd * np.poly([zd, -1])
    A_filt = np.poly([p1d, p2d])
    # MATLAB filter() runs along the first NON-singleton dimension: down the
    # columns of a matrix, along a row vector.  lfilter defaults to axis=-1,
    # which filtered a matrix along its rows instead.  COM Octave, ir_in 4x2
    # = [delta, delta delayed one sample], gives column 1 = the delta response
    # and column 2 = that response delayed by one sample.
    axis = 0 if (ir_in.ndim >= 2 and ir_in.shape[0] != 1) else -1
    impulse_response = lfilter(B_filt, A_filt, ir_in, axis=axis)
    return impulse_response, p1_ctle, p2_ctle, z_ctle


if __name__ == "__main__":
    ir = np.zeros(20)
    ir[0] = 1.0
    ir_out, p1, p2, z = TD_CTLE(ir, 25e9, 5e9, 10e9, 20e9, 0.0, 2)
    print("p1_ctle =", p1, " expected", -2*np.pi*10e9)
    print("ir_out[:5] =", ir_out[:5])
