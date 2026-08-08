import numpy as np
from scipy.signal import lfilter


def read_PR_files(param, OP, chdata):
    """Read pulse response files (MATLAB lines 9569-9600).

    Only .csv extension is supported. Prepends 3*M zeros as precursor guard.
    Returns (chdata, param).
    """
    M = int(param.samples_per_ui)
    for i, cd in enumerate(chdata):
        if cd.ext.lower() == '.csv':
            vt = np.loadtxt(cd.filename)
            t_col = vt[:, 0]
            v_col = vt[:, 1]
            dt = float(t_col[1] - t_col[0])
            guard = 3 * M
            upr = np.concatenate([np.zeros(guard), v_col])
            t = np.concatenate([np.arange(guard) * dt, t_col + guard * dt])
            cd.uneq_pulse_response = upr
            cd.t = t
            N = int(np.floor(len(upr) / M))
            unit = np.concatenate([[1.0], np.zeros(M - 1)])
            step_sv = np.tile(unit, N)
            step_resp = lfilter(step_sv, [1.0], upr)
            cd.uneq_imp_response = step_resp - np.concatenate([[0.0], step_resp[:-1]])
            cd.uneq_imp_response[0] = cd.uneq_imp_response[1]

    return chdata, param
