import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace

_EPS = np.finfo(float).eps


def _colon(step, limit):
    """MATLAB `0:step:limit`.

    np.arange(0, limit + step, step) is one element too long whenever
    limit/step is not an integer, which it is not for the shipped default
    TR_TDR = 8e-3 ns.  COM Octave, fb=106.25e9, samples_per_ui=32,
    TR_TDR=8e-3: numel(0:dt:edge_time*2) is 55, arange gave 56.  Plain
    floor() is not enough either -- it is one short whenever the quotient
    lands a fraction of an eps below an integer, e.g. fb=106.25e9,
    samples_per_ui=32, TR_TDR=0.5325 where the quotient is 3620.9999999999995
    and Octave returns 3622 elements.  The last element is clamped to the
    limit when it would overshoot it, as Octave's range::final_value does.
    """
    n = int(round(limit / step + 1.0))
    if n > 0 and (n - 1) * step > limit + 3.0 * _EPS * abs(limit):
        n -= 1
    out = np.arange(max(n, 0)) * step
    if out.size:
        out[0] = 0.0        # the base, not 0*step, which is NaN for step=Inf
    if out.size > 1 and out[-1] > limit:
        out[-1] = limit
    return out


def _ones_row(n):
    """MATLAB `ones(1,n)`: n must be an integer value, and n<=0 gives empty."""
    if float(n) != int(n):
        # COM Octave: ones(1,2.5) -> "conversion of 2.5 to int64_t value failed"
        raise ValueError(
            'param.samples_per_ui must be an integer, got %r' % (n,))
    return np.ones(max(int(n), 0))


def _filter(b, x):
    """MATLAB `filter(b,1,x)`; scipy's lfilter refuses the empty cases."""
    # COM Octave: filter(ones(1,4),1,zeros(1,0)) -> 1x0, and
    #             filter(zeros(1,0),1,x)         -> zeros(size(x)).
    if x.size == 0:
        return np.zeros(0)
    if b.size == 0:
        return np.zeros(x.shape)
    return lfilter(b, [1.0], x)


def get_PulseR(ir, param, cb_step, ZT):
    """Compute TDR pulse response from impulse response (MATLAB lines 6655-6682).

    cb_step=True: drive with a shaped edge; cb_step=False: rectangular UI pulse.
    Returns SimpleNamespace with fields: PDR (TDR response), pulse (filtered IR).
    """
    ir = np.asarray(ir, dtype=float).ravel()
    M = param.samples_per_ui

    if cb_step:
        # TR_TDR=0 makes fedge Inf and the cosine argument 0*Inf = NaN, and
        # samples_per_ui=0 makes dt Inf so tedge collapses to the single base
        # point.  MATLAB carries both through rather than raising, so these
        # divisions must be numpy's and not Python's.
        with np.errstate(divide='ignore', invalid='ignore'):
            dt = np.float64(1.0) / np.float64(param.fb) / np.float64(M)
            edge_time = float(param.TR_TDR) * 1e-9
            fedge = np.float64(1.0) / np.float64(edge_time)
            tedge = _colon(dt, edge_time * 2)
            edge = 2 * np.cos(2 * np.pi * tedge * fedge / 16 - np.pi / 4) ** 2 - 1
        drive_pulse = np.concatenate([edge, _ones_row(M)])
        pulse = _filter(drive_pulse, ir)
    else:
        pulse = _filter(_ones_row(M), ir)

    PDR_response = (1 + pulse) / (1 - pulse) * float(ZT) * 2
    return SimpleNamespace(PDR=PDR_response, pulse=pulse)
