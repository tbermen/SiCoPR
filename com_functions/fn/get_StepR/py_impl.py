import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


_EPS = np.finfo(float).eps


def _colon(step, limit):
    """MATLAB `0:step:limit`.

    Two wrong spellings were tried here first, and both are recorded because
    each looked verified:

    `np.arange(0, limit + step, step)` is one element too LONG whenever
    limit/step is not an integer, which it is not at the shipped default
    TR_TDR = 8e-3 ns. COM Octave, fb=53.125e9, samples_per_ui=32: numel 28,
    not 29.

    `int(floor(limit/step)) + 1` is one element too SHORT whenever the
    quotient lands a fraction of an eps below an integer. fb=106.25e9,
    samples_per_ui=32, TR_TDR=0.5325 gives 3620.9999999999995, about 2000 eps
    below 3621, and Octave returns 3622 points. That spelling was accepted on
    a 660-combination sweep that compared only numel, and numel is exactly
    what it gets wrong here; a sweep over element VALUES, 672 combinations,
    caught it.

    The last element is clamped to the limit when accumulation would overshoot
    it, as Octave's range::final_value does. Identical to the helper in
    get_PulseR, which is the same reference line.
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


def get_StepR(ir, param, cb_step, ZT):
    """Compute TDR step response from impulse response (MATLAB lines 6876-6902).

    cb_step=True: shaped edge drive; cb_step=False: cumulative sum (ideal step).
    Returns SimpleNamespace with fields: ZSR (TDR response in Ω), pulse (step signal).
    """
    # No dtype=float: MATLAB's filter()/cumsum() carry a complex input
    # through, and the cast silently DISCARDED the imaginary part (a
    # ComplexWarning only).  COM Octave, cb_step=0 on a complex ir, returns
    # cumsum of the complex samples; the cast made ZSR wrong by O(1).
    ir = np.asarray(ir).ravel()
    if ir.dtype.kind not in 'fc':
        ir = ir.astype(float)
    M = int(param.samples_per_ui)

    if cb_step:
        dt = 1.0 / float(param.fb) / float(param.samples_per_ui)
        edge_time = float(param.TR_TDR) * 1e-9
        fedge = 1.0 / edge_time
        # MATLAB `tedge=0:dt:edge_time*2`, via the shared spelling. See
        # _colon below for why neither arange-with-limit nor floor is right.
        tedge = _colon(dt, edge_time * 2)
        edge = 2 * np.cos(2 * np.pi * tedge * fedge / 16 - np.pi / 4) ** 2 - 1
        drive_pulse = np.concatenate([edge, np.ones(M)])
        pulse = lfilter(drive_pulse, [1.0], ir)
    else:
        pulse = np.cumsum(ir)

    TDR_response = (1 + pulse) / (1 - pulse) * float(ZT) * 2
    return SimpleNamespace(ZSR=TDR_response, pulse=pulse)
