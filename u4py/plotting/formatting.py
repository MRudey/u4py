"""
Contains functions for consistent figure and axis formatting.
"""
import re
import string
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mptransf
import numpy as np
import pyproj
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_map_label(text: str, coords: tuple, ax: Axes):
    """Adds a label at the specified coordinates to the axes.

    :param text: The text.
    :type text: str
    :param coords: The coordinates where to label.
    :type coords: tuple
    :param ax: The axes to label.
    :type ax: Axes
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


def map_style(ax: Axes, divisor: int = 0, grid: bool = True, crs: str = ""):
    """Changes axis to be in a good format for a map.

    :param ax: The axis object containing the map.
    :type ax: Axes
    :param divisor: The tick divisor, defaults to 0
    :type divisor: int, optional
    :param grid: Plot a red grid, defaults to True
    :type grid: bool, optional
    :param crs: If given, adds the name of the coordinate system to the map, defaults to ""
    :type crs: str, optional
    """
    if grid:
        ax.grid("True", color="r", alpha=0.3)
    try:
        ax.ticklabel_format(style="plain")
    except AttributeError:
        pass

    if divisor:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(divisor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(divisor))
    ax.xaxis.set_major_formatter(coordinate_formatter)
    ax.yaxis.set_major_formatter(coordinate_formatter)
    plt.yticks(verticalalignment="center", rotation=90)
    if crs:
        crs_obj = pyproj.CRS.from_user_input(crs)
        ax.annotate(
            crs_obj.name,
            (1, -0.1),
            xycoords="axes fraction",
            horizontalalignment="right",
            fontsize="small",
            annotation_clip=False,
        )


def coordinate_formatter(x: float, pos: int) -> str:
    """An axis formatter for UTM style coordinates.

    :param x: The value to be formatted.
    :type x: float
    :param pos: The position of the number.
    :type pos: int
    :return: The formatted number.
    :rtype: str
    """
    pre = str(int(x // 1000))
    post = "%03i" % (int(x % 1000))
    # if post == "000":
    #     post = "0"
    out = "$" + pre[:-1] + "^{" + pre[-1] + post + "}$"
    return out


def numerate_axes(fig: Figure, n: int = 0, step: int = 1):
    """Adds alphabetic numbering to all axes in a figure.

    :param fig: The Figure containing the axes.
    :type fig: Figure
    :param n: starting index, defaults to 0
    :type n: int, optional
    :param step: step index, defaults to 1
    :type step: int, optional
    """
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

    :param text: The text to add
    :type text: str
    :param ax: The axis where to add the text.
    :type ax: Axes
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
    """Generates a drop shape for plotting.

    :return: A `PathPatch` that can be used as a marker.
    :rtype: mpatches.PathPatch
    """
    svg_code = (  # SVG code from freesvg.art
        "M640.552,262.073c-0.04,37.362-31.052,73.551-77.44,80.159c-30.523,"
        + "4.347-55.842-7.202-76.129-29.801c-13.945-15.534-21.116-34.331"
        + "-22.04-54.937c-0.804-17.926,4.137-35.137,10.376-51.886c16.81"
        + "-45.132,41.643-85.667,71.012-123.59c6.084-7.856,6.993-7.717,"
        + "13.018-0.02c28.645,36.595,52.818,75.81,69.581,119.295C635.404,"
        + "218.089,640.942,235.193,640.552,262.073z"
    )
    codes, verts = _svg_parse(svg_code)
    centroid = np.mean(verts, axis=0)
    verts[:, 0] = verts[:, 0] - centroid[0]
    verts[:, 1] = verts[:, 1] - centroid[1]
    verts = verts / np.max(verts)
    path = mpath.Path(verts, codes)
    path = path.transformed(mptransf.Affine2D().rotate_deg(180))
    # patch = mpatches.PathPatch(path, facecolor="r", alpha=0.5)
    return path


def _svg_parse(svg_code: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parses a simple svg string into codes and vertices for matplotlib paths.

    *Adapted from the matplotlib documentation.*

    :param svg_code: A svg string without spaces.
    :type svg_code: str
    :return: Codes and Vertices for a matplotlib path.
    :rtype: Tuple[np.ndarray, np.ndarray]
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
