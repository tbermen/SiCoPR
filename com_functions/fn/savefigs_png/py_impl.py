import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def savefigs_png(param, OP):
    """Save all open matplotlib figures as PNG (MATLAB lines 11216-11255).

    Saves as .png when OP.SAVE_FIGURES==1.
    Saves line data as CSV when OP.SAVE_FIGURE_to_CSV==1.
    Returns list of figure numbers.
    """
    if not _HAS_MPL:
        return []

    fig_nums = plt.get_fignums()
    result_dir = getattr(OP, 'RESULT_DIR', '.')
    runtag = getattr(OP, 'RUNTAG', '')
    base = getattr(param, 'base', '')

    for num in fig_nums:
        fig = plt.figure(num)
        figname = fig.get_label() or ''
        if base and base not in figname:
            figname = f'{figname} {runtag} {base}'.strip()
        figname = f'f_{num}_{figname}'
        figname = figname.replace(':', '-').replace(' ', '_')

        if getattr(OP, 'SAVE_FIGURES', 0) == 1:
            fig.savefig(os.path.join(result_dir, figname + '.png'))

        if getattr(OP, 'SAVE_FIGURE_to_CSV', 0) == 1:
            rows = []
            for ax in fig.get_axes():
                for line in ax.get_lines():
                    rows.append(line.get_xdata())
                    rows.append(line.get_ydata())
            if rows:
                M = np.array(rows)
                np.savetxt(os.path.join(result_dir, figname + '.csv'), M, delimiter=',')

    return fig_nums
