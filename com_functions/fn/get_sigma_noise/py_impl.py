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
    hit = faxis >= float(param.fb) / 2
    if not hit.any():
        # find() is empty, so MATLAB averages over H_np(1:[]) — an empty
        # slice — and mean([]) is NaN.  np.argmax answers 0 on an all-False
        # mask, which averaged over the *first* bin instead and returned a
        # finite number.
        # COM Octave 4p16p0: faxis = linspace(0,fb/4,51), fb=100e9 gives
        # sigma_NE = NaN and sigma_HP = NaN; Python returned 0 and 0.
        return float('nan'), float('nan')
    idx0 = int(np.argmax(hit))
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
    # MATLAB squares first and takes the modulus after: abs(H(1:n).^2).
    # |z^2| and |z|^2 agree mathematically but not bit-for-bit — 18 of the 26
    # in-band elements differ in the probe pinned in test_verify.py, where the
    # reference order makes the f_hp=0 result match COM Octave 4p16p0 exactly.
    sigma_NE = float(sigma_bn) * np.sqrt(np.mean(np.abs(H_np[:idxfbby2] ** 2)))
    sigma_HP = float(sigma_bn) * np.mean(np.abs(H_hp[:idxfbby2] ** 2))
    return sigma_NE, sigma_HP
