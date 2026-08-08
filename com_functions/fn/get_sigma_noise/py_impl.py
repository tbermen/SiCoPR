import numpy as np


def get_sigma_noise(H_ctf, param, chdata, sigma_bn):
    """Compute sigma_NE and sigma_HP for RX calibration noise.

    chdata is a list/array of channel structs; chdata[1] (MATLAB chdata(2)) is used.
    All index arithmetic matches MATLAB find(...,1,'first') 1-based → Python argmax 0-based.
    """
    H_ctf = np.asarray(H_ctf, dtype=complex)
    faxis = np.asarray(chdata[1].faxis, dtype=float).ravel()
    H_r = 1.0 / np.polyval(
        [1, 2.613126, 3.414214, 2.613126, 1],
        1j * faxis / (float(param.f_r) * float(param.fb))
    )
    # MATLAB: idxfbby2 = find(faxis >= fb/2, 1)  [1-based]
    # Python: 0-based argmax → slice [:idxfbby2+1] matches MATLAB (1:idxfbby2)
    idx0 = int(np.argmax(faxis >= float(param.fb) / 2))
    idxfbby2 = idx0 + 1  # exclusive upper bound for slices below

    if len(chdata) >= 2:
        Hnoise_channel = np.asarray(chdata[1].sdd21, dtype=complex).ravel()
    else:
        Hnoise_channel = np.ones(len(faxis), dtype=complex)

    f = faxis
    f_hp = float(param.f_hp)
    if f_hp != 0:
        H_hp = (-1j * f / f_hp) / (1 + 1j * f / f_hp)
    else:
        H_hp = np.ones(len(f), dtype=complex)

    H_np = Hnoise_channel * H_ctf * H_r * H_hp
    sigma_NE = float(sigma_bn) * np.sqrt(np.mean(np.abs(H_np[:idxfbby2]) ** 2))
    sigma_HP = float(sigma_bn) * np.mean(np.abs(H_hp[:idxfbby2]) ** 2)
    return sigma_NE, sigma_HP
