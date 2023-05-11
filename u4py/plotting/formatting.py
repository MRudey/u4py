"""
Contains functions for consistent figure and axis formatting.
"""
import re
import string
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.transforms as mptransf
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


def map_style(ax: Axes, divisor: int = 0, grid: bool = True):
    """Changes axis to be in a good format for a map.

    Arguments:
        ax -- The axis object containing the map.

    Keyword Arguments:
        divisor -- The tick divisor (default: {0}).
        grid -- Plot a red grid (default: {True}).
    """
    if grid:
        ax.grid("True", color="r", alpha=0.3)
    try:
        ax.ticklabel_format(style="plain")
    except AttributeError:
        pass

    if divisor:
        ticks = ax.get_yticks()
        new_ticks = [yt for yt in ticks if np.remainder(yt, divisor) == 0]
        ax.set_yticks(new_ticks)

        ticks = ax.get_xticks()
        new_ticks = [yt for yt in ticks if np.remainder(yt, divisor) == 0]
        ax.set_xticks(new_ticks)
    else:
        ax.set_yticks(ax.get_yticks())

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


def add_copyright(text: str, ax: Axes):
    """Adds a small copyright string to the lower left of the plot.

    Arguments:
        text -- The text to add
        ax -- The axis where to add the text.
    """

    txt = ax.annotate(
        text,
        (0.02, 0.02),
        xycoords="axes fraction",
        fontsize="small",
        fontstyle="italic",
    )
    txt.set_path_effects(
        [
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ]
    )


def drop_shape() -> mpatches.PathPatch:
    """Generates a drop shape for plotting

    Returns:
        A `PathPatch` that can be used as a marker.
    """
    svg_code = (  # SVG code from freesvg.art
        "M640.552,262.073c-0.04,37.362-31.052,73.551-77.44,80.159c-30.523,"
        + "4.347-55.842-7.202-76.129-29.801c-13.945-15.534-21.116-34.331"
        + "-22.04-54.937c-0.804-17.926,4.137-35.137,10.376-51.886c16.81"
        + "-45.132,41.643-85.667,71.012-123.59c6.084-7.856,6.993-7.717,"
        + "13.018-0.02c28.645,36.595,52.818,75.81,69.581,119.295C635.404,"
        + "218.089,640.942,235.193,640.552,262.073z"
    )
    codes, verts = svg_parse(svg_code)
    centroid = np.mean(verts, axis=0)
    verts[:, 0] = verts[:, 0] - centroid[0]
    verts[:, 1] = verts[:, 1] - centroid[1]
    verts = verts / np.max(verts)
    path = mpath.Path(verts, codes)
    path = path.transformed(mptransf.Affine2D().rotate_deg(180))
    # patch = mpatches.PathPatch(path, facecolor="r", alpha=0.5)
    return path


def svg_parse(svg_code: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parses a simple svg string into codes and vertices for matplotlib paths.

    Adapted from the matplotlib documentation.

    Arguments:
        svg_code -- A svg string without spaces

    Returns:
        Codes and Vertices for a matplotlib path
    """
    commands = {
        "M": (mpath.Path.MOVETO,),
        "L": (mpath.Path.LINETO,),
        "Q": (mpath.Path.CURVE3,) * 2,
        "C": (mpath.Path.CURVE4,) * 3,
        "Z": (mpath.Path.CLOSEPOLY,),
    }
    vertices = []
    codes = []
    cmd_values = re.split("([A-Za-z])", svg_code)[1:]  # Split over commands.
    for cmd, values in zip(cmd_values[::2], cmd_values[1::2]):
        # Numbers are separated either by commas, or by +/- signs (but not at
        # the beginning of the string).
        points = (
            [*map(float, re.split(",|(?<!^)(?=[+-])", values))]
            if values
            else [(0.0, 0.0)]
        )  # Only for "z/Z" (CLOSEPOLY).
        points = np.reshape(points, (-1, 2))
        if cmd.islower():
            points += vertices[-1][-1]
        codes.extend(commands[cmd.upper()])
        vertices.append(points)
    return np.array(codes), np.concatenate(vertices)
