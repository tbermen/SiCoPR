import io

import numpy as np
from scipy.signal import lfilter


def read_PR_files(param, OP, chdata):
    """Read pulse response files (MATLAB lines 9569-9600).

    Only .csv extension is supported. Prepends 3*M zeros as precursor guard.
    Returns (chdata, param).
    """
    M = int(param.samples_per_ui)
    for i, cd in enumerate(chdata):
        # MATLAB's `switch chdata(i).ext; case '.csv'` is CASE SENSITIVE, so a
        # file named .CSV falls through and the channel keeps no pulse
        # response at all.  COM Octave 4p16p0 with ext '.CSV': chdata comes
        # back with only filename and ext.  .lower() read it instead.
        if cd.ext == '.csv':
            # MATLAB reads it with load(), which takes space, comma OR tab as
            # the delimiter -- and the extension is .csv, so commas are the
            # expected case.  np.loadtxt splits on whitespace only and stopped
            # with "could not convert string '0,4.98910939279e-20' to
            # float64" on a genuinely comma-separated file that COM Octave
            # read without complaint.
            with open(cd.filename, 'r') as fh:
                text = fh.read().replace(',', ' ')
            vt = np.loadtxt(io.StringIO(text))
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
