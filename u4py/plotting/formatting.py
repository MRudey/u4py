"""
Contains functions for consistent figure and axis formatting.
"""

from __future__ import annotations

import os
import pickle as pkl
import re
import string
import textwrap
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
        zorder=25,
    )
    outline_text(txt)


def outline_text(txt: mpath.Path, color: str = "white", width: float = 3):
    """Helper function to format text with outline.

    :param txt: The text object.
    :type txt: str
    :param color: The outline color, defaults to "white"
    :type color: str, optional
    :param width: The outline width, defaults to 3
    :type width: float, optional
    """
    txt.set_path_effects(
        [
            path_effects.Stroke(linewidth=width, foreground=color),
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
    ax.set_yticks(
        ax.get_yticks(),
        ax.get_yticklabels(),
        verticalalignment="center",
        rotation=90,
    )
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
        xlbl, ylbl = get_axislabel_from_crs(crs_obj)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)


def get_axislabel_from_crs(crs: pyproj.CRS) -> Tuple[str]:
    """Gets the information from the crs object and creates appropriate axis labels.

    :param crs: The coordinate system for the axis.
    :type crs: pyproj.CRS
    """

    xname = crs.axis_info[0].name
    yname = crs.axis_info[1].name
    xunit = crs.axis_info[0].unit_name
    yunit = crs.axis_info[0].unit_name

    xname = xname.replace("Geodetic l", "L")
    yname = yname.replace("Geodetic l", "L")
    if xunit == "degree":
        xunit = "°"
    elif xunit == "metre":
        xunit = "m"

    if yunit == "degree":
        yunit = "°"
    elif yunit == "metre":
        yunit = "m"

    xstr = f"{xname} ({xunit})"
    ystr = f"{yname} ({yunit})"
    return (xstr, ystr)


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


def enumerate_axes(fig: Figure, n: int = 0, step: int = 1, ignore: list = []):
    """Adds alphabetic numbering to all axes in a figure.

    :param fig: The Figure containing the axes.
    :type fig: Figure
    :param n: starting index, defaults to 0
    :type n: int, optional
    :param step: step index, defaults to 1
    :type step: int, optional
    :param ignore: axes to ignore, defaults to []
    :type ignore: list, optional

    """
    axes = fig.get_axes()

    for ii in range(0, len(axes), step):
        if ii not in ignore:
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


def add_copyright(text: str, ax: Axes, textpos: tuple = (0.02, 0.02)):
    """Adds a small copyright string to the lower left of the plot.

    :param text: The text to add
    :type text: str
    :param ax: The axis where to add the text.
    :type ax: Axes
    """
    txt = ax.annotate(
        text,
        textpos,
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


def add_scalebar(
    ax: Axes,
    width: float = 1000,
    unit: str = "m",
    div: int = 2,
    loc: str = "bottom left",
    limit_width: float = 0.5,
):
    """Adds a scalebar to the given axis.

    If `limit_width` is set (default behaviour) the width of the scalebar is
    automatically set to occupy a maximum of `limit_width` axes units. To
    override this behaviour set `limit_width=0`.

    :param ax: The axis to add the data to.
    :type ax: Axes
    :param width: The full scale width, defaults to 1000
    :type width: float, optional
    :param unit: The unit of the width, defaults to "m"
    :type unit: str, optional
    :param div: The number of subdivisions to create (always halves the first division `|---|---|------|`), defaults to 2
    :type div: int, optional
    :param loc: Location of the scale bar, same syntax as the matplotlib legend placement, defaults to "bottom left"
    :type loc: str, optional
    :param limit_width: Maximum width of the scale bar in axes units (0..1), defaults to 0.5
    :type limit_width: float, optional

    """

    if (
        ("left" not in loc)
        and ("right" not in loc)
        and ("top" not in loc)
        and ("bottom" not in loc)
    ):
        raise ValueError(
            "Invalid location string, allowed values are: 'top', 'bottom', "
            + "'left', 'right', and combinations thereof."
        )

    if "right" in loc:
        xf = 0.95
    elif "left" in loc:
        xf = 0.05
    else:
        xf = 0.5

    if "top" in loc:
        yf = 0.95
    elif "bottom" in loc:
        yf = 0.05
    else:
        yf = 0.5

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    cur_width = xlim[1] - xlim[0]
    x = xlim[0] + cur_width * xf
    y = ylim[0] + (ylim[1] - ylim[0]) * yf
    height = cur_width * 0.025
    if not width or (limit_width and cur_width / width > limit_width):
        axw = cur_width * limit_width
        widths = [
            10000,
            7500,
            5000,
            2500,
            1000,
            750,
            500,
            250,
            100,
            75,
            50,
            25,
            10,
        ]
        for w in widths:
            if axw >= w:
                width = w
                break
            else:
                width = 10

    ax.annotate(
        unit,
        (x + width + (0.01 * width), y - (height * 0.2)),
        horizontalalignment="left",
        verticalalignment="bottom",
        # fontsize="large",
        fontweight="bold",
        color="w",
        path_effects=[path_effects.withStroke(linewidth=2, foreground="k")],
        zorder=99,
    )

    for ii in range(div):
        ax.add_patch(
            mpatches.Rectangle(
                (x, y), width, height, facecolor="w", edgecolor="k", zorder=99
            )
        )
        _label_scalebar(ax, x, y, width, height)
        ax.add_patch(
            mpatches.Rectangle(
                (x, y),
                width / 2,
                height,
                facecolor="k",
                edgecolor="k",
                zorder=99,
            )
        )
        if ii < div - 1:
            _label_scalebar(ax, x, y, width / 2, height)
        else:
            _label_scalebar(ax, x, y, 0, height)

        width /= 4


def _label_scalebar(ax: Axes, x: float, y: float, width: float, height: float):
    """Adds a label to the scalebar at the given position.

    :param ax: The axis for plotting.
    :type ax: Axes
    :param x: The left edge of the scalebar
    :type x: float
    :param y: The bottom edge of the scalebar
    :type y: float
    :param width: The width where to put the label.
    :type width: float
    :param height: The height of the scalebar for positioning.
    :type height: float
    """
    if width % 1 > 0:
        w_str = str(width)
    else:
        w_str = str(int(width))
    ax.annotate(
        w_str,
        (x + width, y + height * 1.15),
        horizontalalignment="center",
        verticalalignment="bottom",
        # fontsize="large",
        fontweight="bold",
        color="w",
        path_effects=[path_effects.withStroke(linewidth=2, foreground="k")],
        zorder=10,
    )


def get_style_from_legend(
    map_uuids: list, legend_path: os.PathLike
) -> Tuple[dict, list]:
    """
    Creates a dictionary with plot format arguments and a list of legend
    handles to plot the geodataframe according to a pickled legend converted
    from the json information of HLNUG.

    :param map_uuids: The unique identifiers of the feature geometries.
    :type map_uuids: list
    :param legend_path: The path to the legend file
    :type legend_path: os.PathLike
    :return: A dictionary with the plotting arguments and a list of legend handles.
    :rtype: Tuple[dict, list]
    """
    with open(legend_path, "rb") as leg_file:
        (
            uuids,
            labels,
            styles,
            fills,
            facecolors,
            edgecolors,
            alphas,
            linewidths,
        ) = pkl.load(leg_file)

    # Get color and style for all map uuids
    idx = [(uuids.index(uuid) if uuid in uuids else -1) for uuid in map_uuids]
    leg_dict = {
        # "lb": [(labels[ii] if ii >= 0 else "unknown") for ii in idx],
        "hatch": [(styles[ii] if ii >= 0 else "") for ii in idx],
        "fill": [(fills[ii] if ii >= 0 else True) for ii in idx],
        "fc": [(facecolors[ii] if ii >= 0 else (0.5, 0.5, 0.5)) for ii in idx],
        "ec": [(edgecolors[ii] if ii >= 0 else (0.5, 0.5, 0.5)) for ii in idx],
        "alpha": [(alphas[ii] if ii >= 0 else 0.5) for ii in idx],
        "linewidth": [(linewidths[ii] if ii >= 0 else 1) for ii in idx],
    }

    # Create legend entries
    un_uuid = np.unique(map_uuids)
    un_uuid.sort()
    leg_handles = []
    leg_labels = []

    for uuid in un_uuid:
        try:
            ii = uuids.index(uuid)
            if isinstance(styles[ii], str):
                leg_handles.append(
                    mpatches.Patch(
                        facecolor=facecolors[ii],
                        edgecolor=edgecolors[ii],
                        alpha=alphas[ii],
                        fill=fills[ii],
                        hatch=styles[ii],
                        linewidth=linewidths[ii],
                    )
                )
            elif isinstance(styles[ii], dict):
                if not styles[ii]["hatch"]:
                    leg_handles.append(
                        mpatches.Patch(
                            facecolor=facecolors[ii],
                            edgecolor=edgecolors[ii],
                            alpha=alphas[ii],
                            fill=fills[ii],
                            linewidth=linewidths[ii],
                        )
                    )
                else:
                    handle = []
                    handle.append(
                        mpatches.Patch(
                            facecolor=facecolors[ii],
                            edgecolor=edgecolors[ii],
                            alpha=alphas[ii],
                            fill=fills[ii],
                            linewidth=linewidths[ii],
                        )
                    )
                    for hii in range(len(styles[ii]["hatch"])):
                        ht = styles[ii]["hatch"][hii]
                        hc = styles[ii]["hatch_color"][0]
                        hca = styles[ii]["hatch_alpha"][0]
                        handle.append(
                            mpatches.Patch(
                                facecolor="None",
                                edgecolor=hc,
                                alpha=hca,
                                hatch=ht,
                                linewidth=linewidths[ii],
                            )
                        )
                    leg_handles.append(tuple(handle))
            txt_width = 80
            leg_labels.append(textwrap.fill(labels[ii], txt_width))

        except ValueError:
            leg_handles.append(
                mpatches.Patch(
                    facecolor=(0.5, 0.5, 0.5),
                    edgecolor=(0.5, 0.5, 0.5),
                    alpha=0.5,
                    fill=True,
                    hatch="",
                    linewidth=1,
                )
            )
            leg_labels.append("unknown")

    return leg_dict, leg_handles, leg_labels


def full_screen_map(ax: Axes, fix_axes: bool = True):
    """Makes the axis fill the whole figure and fixes the current extent.

    :param ax: The axis to make full screen.
    :type ax: Axes
    :param fix_axes: Whether to fix the axis limits as is, defaults to True
    :type fix_axes: bool, optional
    """

    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    plt.axis("equal")
    plt.draw()

    if fix_axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
