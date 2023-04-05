"""
Contains functions for consistent figure and axis formatting.
"""


import string

import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_map_label(text: str, coords: tuple, ax: Axes):
    """Adds a label at the specified coordinates to the axes

    Arguments:
        text -- The text.
        coords -- The coordinates where to label.
        ax -- The axes to label.
    """
    txt = ax.annotate(
        text,
        coords,
        # xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",
    )
    txt.set_path_effects(
        [
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ]
    )


def map_style_ticks(ax: Axes, divisor: int = 2000):
    """Changes the x and y ticks to be in a good format for a map.

    Arguments:
        ax -- The axis object containing the map.

    Keyword Arguments:
        divisor -- The tick divisor (default: {2000}).
    """
    ax.ticklabel_format(style="plain")

    ticks = ax.get_yticks()
    new_ticks = [yt for yt in ticks if np.remainder(yt, divisor) == 0]
    ax.set_yticks(new_ticks)

    ticks = ax.get_xticks()
    new_ticks = [yt for yt in ticks if np.remainder(yt, divisor) == 0]
    ax.set_xticks(new_ticks)

    ax.set_yticklabels(
        ax.get_yticks().astype(int),
        rotation=90,
        verticalalignment="center",
    )


def numerate_axes(fig: Figure, n: int = 0, step: int = 1):
    """Adds numbering to all axes in a figure.

    Arguments:
        fig -- The Figure containing the axes.

    Keyword Arguments:
        n -- starting index (default: {0}).
        step -- step index (default: {1}).
    """ """"""
    axes = fig.get_axes()

    for ii in range(0, len(axes), step):
        axes[ii].annotate(
            "(" + string.ascii_lowercase[int((ii + n) / step)] + ")",
            (-0.1, 1.05),
            xycoords="axes fraction",
            fontweight="bold",
            fontsize="xx-large",
            verticalalignment="center",
            horizontalalignment="center",
            # bbox=dict(fc="w", boxstyle="Circle"),
        )
