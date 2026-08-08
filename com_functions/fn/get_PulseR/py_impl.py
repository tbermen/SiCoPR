import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


def get_PulseR(ir, param, cb_step, ZT):
    """Compute TDR pulse response from impulse response (MATLAB lines 6655-6682).

    cb_step=True: drive with a shaped edge; cb_step=False: rectangular UI pulse.
    Returns SimpleNamespace with fields: PDR (TDR response), pulse (filtered IR).
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
        pulse = lfilter(np.ones(M), [1.0], ir)

    PDR_response = (1 + pulse) / (1 - pulse) * float(ZT) * 2
    return SimpleNamespace(PDR=PDR_response, pulse=pulse)
