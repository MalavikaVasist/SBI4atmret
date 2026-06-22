import numpy as np
import matplotlib as mpl
from lampe.plots import LinearAlphaColormap


def legends(axes=None, creds=(0.6827, 0.9545, 0.9973), alpha=(0.0, 0.5), color="steelblue"):
    """
    Build legend handles including credible-region patches.

    Args:
        axes: matplotlib Axes (single axis or array)
        creds: credibility levels
        alpha: (min, max) alpha range for colormap
        color: base color string

    Returns:
        (handles, texts) lists for ax.legend(handles, texts)
    """

    if np.shape(axes) != ():
        handles, texts = axes[0, -1].get_legend_handles_labels()
    else:
        handles, texts = axes.get_legend_handles_labels()

    creds = np.sort(np.asarray(creds))[::-1]
    creds = np.append(creds, 0)

    cmap = LinearAlphaColormap(color, levels=creds, alpha=alpha)

    levels = (creds - creds.min()) / (creds.max() - creds.min())
    levels = (levels[:-1] + levels[1:]) / 2

    for c, l in zip(creds[:-1], levels):
        handles.append(mpl.patches.Patch(color=cmap(l), linewidth=0))
        texts.append(r"${:.1f}\,\%$ credible region".format(c * 100))

    return handles, texts
