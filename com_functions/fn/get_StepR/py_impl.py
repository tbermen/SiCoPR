import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


def get_StepR(ir, param, cb_step, ZT):
    """Compute TDR step response from impulse response (MATLAB lines 6876-6902).

    cb_step=True: shaped edge drive; cb_step=False: cumulative sum (ideal step).
    Returns SimpleNamespace with fields: ZSR (TDR response in Ω), pulse (step signal).
    """
    ir = np.asarray(ir, dtype=float).ravel()
    M = int(param.samples_per_ui)

    if cb_step:
        dt = 1.0 / float(param.fb) / float(param.samples_per_ui)
        edge_time = float(param.TR_TDR) * 1e-9
        fedge = 1.0 / edge_time
        tedge = np.arange(0, edge_time * 2 + dt, dt)
        edge = 2 * np.cos(2 * np.pi * tedge * fedge / 16 - np.pi / 4) ** 2 - 1
        drive_pulse = np.concatenate([edge, np.ones(M)])
        pulse = lfilter(drive_pulse, [1.0], ir)
    else:
        pulse = np.cumsum(ir)

    TDR_response = (1 + pulse) / (1 - pulse) * float(ZT) * 2
    return SimpleNamespace(ZSR=TDR_response, pulse=pulse)
