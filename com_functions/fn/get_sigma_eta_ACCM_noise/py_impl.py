import numpy as np


def get_sigma_eta_ACCM_noise(chdata, param, H_sy, H_r, H_ctf):
    """Compute combined sigma_N from thermal and AC CM noise (Eq. 93A-35).

    chdata is a list/array of channel structs. H_sy, H_r, H_ctf are 1-D complex arrays.
    eta_0 is V^2/GHz; diff(faxis) is divided by 1e9 to match units.
    """
    H_sy = np.asarray(H_sy, dtype=complex).ravel()
    H_r = np.asarray(H_r, dtype=complex).ravel()
    H_ctf = np.asarray(H_ctf, dtype=complex).ravel()
    faxis = np.asarray(chdata[0].faxis, dtype=float).ravel()
    df = np.diff(faxis) / 1e9  # convert Hz steps to GHz for eta_0 units

    sigma_N1 = np.sqrt(
        float(param.eta_0)
        * np.sum(np.abs(H_sy[1:] * H_r[1:] * H_ctf[1:]) ** 2 * df)
    )

    if np.sum(np.asarray(param.AC_CM_RMS)) != 0:
        sigma_ACCM = 0.0
        f_int = faxis[faxis <= float(param.ACCM_MAX_Freq)]
        nf = len(f_int)
        for i in range(len(chdata)):
            H_dc = np.abs(np.squeeze(np.asarray(chdata[i].sdc21, dtype=complex)))
            sigma_ACCM_acc = np.sqrt(
                2 * float(param.AC_CM_RMS_TX) ** 2
                * np.sum(
                    np.abs(H_sy[1:nf] * H_r[1:nf] * H_ctf[1:nf] * H_dc[1:nf]) ** 2
                    * np.diff(f_int)
                )
                / float(f_int[-1])
            )
            sigma_ACCM = float(np.linalg.norm([sigma_ACCM_acc, sigma_ACCM]))
        sigma_N = float(np.linalg.norm([sigma_N1, sigma_ACCM]))
    else:
        sigma_N = sigma_N1

    return sigma_N
