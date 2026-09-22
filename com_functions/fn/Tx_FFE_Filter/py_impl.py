# ============================================================
# MATLAB→Python translation notes for Tx_FFE_Filter
# MATLAB lines: 4697–4741
# ============================================================
# Uses varargin_extractor internally — inlined here via keyword args
#   (protocol: no cross-module imports from sibling py_impl.py files).
# isempty(param): param is None → use defaults
# ~isfield(param,'Pkg_TXFFE_preset') → not hasattr(param,'Pkg_TXFFE_preset')
# isempty(f): f is None → create default sweep 0:10e6:fb
#   MATLAB 0:10e6:fb → np.arange(0, fb+10e6, 10e6) to include fb endpoint
# [mcur,icur]=max(Tx_FFE): 1-based in MATLAB; 0-based (argmax) in Python
#   Crucially, (ii_matlab - icur_matlab) == (ii_python - icur_python) because
#   both shift by the same offset → formula is unchanged.
# 1-based loop ii=1..N → Python enumerate (ii, c) with 0-based ii
# ============================================================

import numpy as np


def Tx_FFE_Filter(param=None, f=None, Use_Tx_FFE=None):
    if Use_Tx_FFE is None:
        Use_Tx_FFE = 0

    if param is None:
        fb = 106.25e9
        Tx_FFE = np.array([1.0])
    else:
        fb = param.fb
        if not hasattr(param, 'Pkg_TXFFE_preset'):
            Tx_FFE = np.array([1.0])
        else:
            Tx_FFE = np.asarray(param.Pkg_TXFFE_preset, dtype=float)

    if f is None:
        # MATLAB f=0:10e6:param.fb stops at or below fb. np.arange(0, fb+10e6, 10e6)
        # overshoots by one point whenever fb is not a multiple of 10e6.
        # COM Octave, fb=53.125e9: numel(0:10e6:fb) == 5313, np.arange gave 5314.
        f = np.arange(int(np.floor(fb / 10e6)) + 1) * 10e6
    else:
        f = np.asarray(f, dtype=float)

    if Use_Tx_FFE != 0:
        H_TxFFE = np.zeros(len(f), dtype=complex)
        if Tx_FFE.size:
            # MATLAB max([]) returns an empty icur and `for ii=1:0` runs zero
            # times, so an empty preset leaves H_TxFFE at zeros(1,length(f)).
            # np.argmax([]) raises instead.  COM Octave, Pkg_TXFFE_preset=[]:
            # H_TxFFE == [0 0 0].
            icur = int(np.argmax(Tx_FFE))      # 0-based; mirrors MATLAB 1-based icur
            for ii, c in enumerate(Tx_FFE):    # ii 0-based → (ii-icur) == MATLAB (ii-icur)
                H_TxFFE += c * np.exp(-1j * 2 * np.pi * (ii - icur) * f / fb)
    else:
        H_TxFFE = np.ones(len(f))

    return H_TxFFE


if __name__ == "__main__":
    from types import SimpleNamespace
    p = SimpleNamespace(fb=25e9, Pkg_TXFFE_preset=[0.5, 1.0, -0.2])
    H = Tx_FFE_Filter(param=p, f=np.array([0.0, 12.5e9, 25e9]), Use_Tx_FFE=1)
    print("H at [0, fb/2, fb]:", H)
