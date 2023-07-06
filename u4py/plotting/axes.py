"""
Contains functions to create simple axes objects to add to other plots. The plots generated here can be used as stand-alone small plots or to compile larger more complicated plots. Because of this, we use only a simple, minimalistic input structure. Each function only should plot a single thing and do only minimal processing. Usually, for a new plot, one would go into :func:`u4py.plotting.plots` and create a new plot function there. Each axis added to the larger plot should be implemented to :func:`u4py.plotting.axes`. This allows also to reuse individual subplots for other figures and also for having a stand-alone version of the axis as a small simple plot.

Internally we use a decorator (`@_add_or_create`) to check if an axis object was given and to generate figure and axis if necessary. Therefore, we have to follow the function template in order for the decorator to work correctly.

All functions should follow the following template::

    plot_FUNCNAME(data: dict, ax: Axes=None, **kwargs) -> None, Tuple[Figure, Axes]

- Functions may also be used to add the plot to an existing plot. If given an `Axes` the function will plot into the given axis (useful for subplots). If no `Axes` is given the function will create a Figure and return both, an `Axes` and `Figure` object.
- The input `data` is required but can also be split into seperate variables.
- For flexibility keywords can be added, e.g. to change plot color.

"""
from __future__ import annotations

import os
from functools import wraps
from typing import Callable, Tuple

import contextily
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rioplot
import scipy.stats as spstats
import skimage.transform as sktransf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyproj import CRS

import u4py.analysis.other as u4other
import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def _add_or_create(internal_plot: Callable) -> Tuple[Figure, Axes] | None:
    """Decorator for all functions in this module

    Functions decorated can either add their plot to the axis (if `ax` is set).
    Otherwise the decorator creates a new plot with `plt.subplots`

    :param internal_plot: The function creating the plot.
    :type internal_plot: Callable
    :return: The outputs of the plotting function.
    :rtype: Tuple[Figure, Axes] | None
    """

    @wraps(internal_plot)
    def wrapper_internal_plot(*args, ax=None, **kwargs):
        # Plotting
        if not ax:
            fig, ax = plt.subplots()
            plot_return = internal_plot(*args, ax=ax, **kwargs)
            if plot_return:
                return fig, ax, plot_return
            else:
                return fig, ax
        else:
            plot_return = internal_plot(*args, ax=ax, **kwargs)
            if plot_return:
                return plot_return
            else:
                return

    return wrapper_internal_plot


@_add_or_create
def plot_stat_func(
    data: dict, ax: Axes, stat_fnc: Callable = np.nanmean
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the given statistic.

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param stat_fnc: The statistical function to use on the timeseries data., defaults to np.nanmean
    :type stat_fnc: Callable, optional
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    time = data["time"]
    y = stat_fnc(data["timeseries"], axis=0)
    ax.plot(time, y)


@_add_or_create
def plot_cwt(data: dict, ax: Axes) -> Tuple[Figure, Axes] | None:
    """Does a continous wavelet transform of the data to find dominant frequencies and plots it.

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """

    time = data["time"]
    y = np.median(data["timeseries"], axis=0)
    freqs, coi, power = u4other.cwt(y, 6)
    xx, yy = np.meshgrid(time, freqs)
    ax.contourf(xx, yy, power, levels=20)
    ax.fill_between(time, coi, np.min(freqs), color="w", alpha=0.25)


@_add_or_create
def plot_pdf(
    y: np.ndarray, ax: Axes, fnc: spstats.rv_continuous = spstats.norm
) -> (
    spstats.distributions.rv_frozen
    | Tuple[spstats.distributions.rv_frozen, Figure, Axes]
):
    """Adds a plot of the probability density function (WIP).

    :param y: The data to be used to fit the distribution.
    :type y: np.ndarray
    :param ax: The axis to add the plot to (optional)
    :type ax: Axes
    :param fnc: Distribution class used to fit the data, defaults to spstats.norm
    :type fnc: spstats.rv_continuous, optional
    :raises NotImplementedError: Currently only returns an error!
    :return: Returns the frozen distribution object or a tuple with object, figure and axis.
    :rtype: spstats.distributions.rv_frozen | Tuple[spstats.distributions.rv_frozen, Figure, Axes]
    """
    raise NotImplementedError("Implement the return of stat vals...")
    # xlims = ax.get_xlim()
    # ax.hist(y, "auto", density=True)
    # t_x = np.linspace(xlims[0], xlims[1], 100)
    # stat_vals = fnc.fit(y)
    # ax.plot(t_x, fnc.pdf(t_x, *stat_vals))


@_add_or_create
def plot_timeseries(
    x: np.ndarray, y: np.ndarray, ax: Axes, color: str = "C0"
) -> Tuple[Figure, Axes] | None:
    """Creates a nice plot of a timeseries including the range of values.

    :param x: 1D Array containing the data for the x-axis.
    :type x: np.ndarray
    :param y: 1D or 2D Array containing the data for the y-axis.
    :type y: np.ndarray
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param color: The color for the plot, defaults to "C0"
    :type color: str, optional
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """

    quantiles = u4plotprep.get_timeseries_range(y)
    plot_quantile_timeseries(x, quantiles, ax=ax, color=color)


@_add_or_create
def plot_quantile_timeseries(
    x: np.ndarray, quantiles: dict, ax: Axes, color: str = "C0"
) -> Tuple[Figure, Axes] | None:
    """Creates a nice plot of a timeseries including the range of values.

    :param x: 1D Array containing the data for the x-axis.
    :type x: np.ndarray
    :param quantiles: Dictionary containing the quantiles for plotting
    :type quantiles: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param color:  The color for the plot, defaults to "C0"
    :type color: str, optional
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    ax.plot(x, quantiles["y_med"], ".", label="Median", color=color)
    ax.fill_between(
        x,
        quantiles["y_95"],
        quantiles["y_5"],
        color=color,
        alpha=0.5,
        edgecolor=None,
        label="Data Range",
    )
    ax.fill_between(
        x,
        quantiles["y_68"],
        quantiles["y_32"],
        color=color,
        alpha=0.5,
        edgecolor=None,
    )


@_add_or_create
def plot_timeseries_fit(
    ax: Axes,
    results: dict = dict(),
    color: str = "C1",
) -> Tuple[Figure, Axes] | None:
    """Plots the fit data for a simple timeseries analysis.

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param color: The color for the plot, defaults to "C0"
    :type color: str, optional
    :param results: A dictionary of loaded fit results, defaults to dict()
    :type results: dict, optional
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    if "signals" in results.keys():
        ax.plot(
            results["time"],
            results["signals"]["lin"] + results["signals"]["sin"],
            label="Simple Fit",
            color=color,
        )
        # shift = time[0] - timedelta(days=sin_popt[2])

        # if shift.day > 10:
        #     prefix = "Mid"
        # elif shift.day > 20:
        #     prefix = "End"
        # else:
        #     prefix = "Start"
        ax.annotate(
            "Longterm Trend: %.1f mm/a" % (results["fits"]["lin"][0] * 365)
            + "\nYearly Variation: $\\pm$%.1f mm"
            % (np.abs(results["fits"]["sin"][0])),
            # + "\nPeriod: %i days" % ((1 / sin_popt[1]) * 6)
            # + "\nMaximum: %s %s" % (prefix, shift.strftime("%B")(0.99, 0.01)),
            (0.99, 0.01),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    else:
        quantiles = u4plotprep.get_timeseries_range(results["U"], results["t"])
        plot_quantile_timeseries(
            quantiles["t_u"], quantiles, ax=ax, color="C0"
        )
        t_fly = u4convert.get_floatyear(quantiles["t_u"])
        t_q = np.linspace(np.min(t_fly), np.max(t_fly), 200)

        y_fit = u4plotprep._full_inv_single_comp(
            t_q,
            *results["U"]["inversion_results"],
        )
        ax.plot(
            u4convert.get_datetime(t_q), y_fit, color=color, label="Inversion"
        )
        ax.annotate(
            "Longterm Trend: %.1f mm/a"
            % (results["U"]["inversion_results"][1])
            + "\nYearly Variation: $\\pm$%.1f mm"
            % (np.abs(results["U"]["inversion_results"][2])),
            # + "\nPeriod: %i days" % ((1 / sin_popt[1]) * 6)
            # + "\nMaximum: %s %s" % (prefix, shift.strftime("%B")(0.99, 0.01)),
            (0.99, 0.01),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="bottom",
        )


@_add_or_create
def plot_region_trend(
    data_region: dict, ax: Axes
) -> Tuple[Figure, Axes] | None:
    """Calculates and plots the regional trend in a region of data.

    :param data_region: Dictionary with multiple data dictionaries according to u4py standard (e.g. read from h5).
    :type data_region: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    region_trend = u4plotprep.get_linfit_each_timeseries(data_region)

    rng = np.percentile(np.abs(region_trend), 95)
    sc = ax.scatter(
        data_region["x"],
        data_region["y"],
        c=region_trend,
        vmin=-rng,
        vmax=rng,
        cmap="RdYlBu",
        label="PSI locations",
        s=5,
    )
    plt.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        extend="both",
        label="Mean Vertical Velocity (mm/a)",
    )


@_add_or_create
def add_shapefile(
    shp_path: os.PathLike, ax: Axes, **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot from the shapefile in the given coordinate system.

    :param shp_path: The path to the shapefile containing the geometry.
    :type shp_path: os.PathLike
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param crx: The target coordinate system as accepted by GeoPandas (default: {"EPSG:23032"}).
    :type crx: str
    :param kwargs: Additional arguments passed to shape.plot(). See GeoPandas documentation.
    :type kwargs: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    kwgs = {"crs": "EPSG:23032", "keys": None, "labels": None}
    kwgs.update(kwargs)
    shape = gp.read_file(shp_path)
    if isinstance(kwgs["crs"], str):
        if shape.crs.to_string() != kwgs["crs"]:
            shape = shape.to_crs(kwgs["crs"])
    elif isinstance(kwgs["crs"], CRS):
        shape = shape.to_crs(kwgs["crs"])
    kwgs.pop("crs")

    if kwgs["keys"]:
        for k, l in zip(kwgs["keys"], kwgs["labels"]):
            to_label = shape[shape.name == k]
            ax.annotate(
                l,
                (to_label.geometry.x, to_label.geometry.y),
                xytext=(5, -5),
                textcoords="offset points",
                color=kwgs["color"],
            )

    kwgs.pop("keys")
    kwgs.pop("labels")
    shape.plot(ax=ax, **kwgs)


@_add_or_create
def add_basemap(
    base_map_path: os.PathLike = None, ax: Axes = None, **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the basemap as the lowest layer. If no basemap is given it is automatically loaded from osm.

    :param base_map_path:  Path to the geotiff with the basemap, defaults to None
    :type base_map_path: os.PathLike, optional
    :param ax: The axis to add the plot to (optional)., defaults to None
    :type ax: Axes, optional
    :param kwargs:  Additional arguments passed to rasterio.plot.show(). See rasterio documentation.
    :type kwargs: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    if base_map_path:
        with rasterio.open(base_map_path) as base_map:
            rioplot.show(base_map, ax=ax, zorder=0, **kwargs)
    else:
        contextily.add_basemap(
            ax=ax,
            crs="EPSG:23032",
            source=contextily.providers.OpenStreetMap.Mapnik,
        )


@_add_or_create
def plot_inversion_fit(
    x: np.ndarray,
    inv_results: tuple,
    ax: Axes,
    direction: str = "UD",
    **kwargs,
) -> Tuple[Figure, Axes] | None:
    """Creates a plot from inversion results.

    :param x: The x axis (datetime).
    :type x: np.ndarray
    :param inv_results: A tuple of the inversion results.
    :type inv_results: tuple
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param direction: The direction of the inversion results. Options are: "EW",
        "NS", "UD", defaults to "UD"
    :type direction: str, optional
    :param kwargs: Additional arguments passed to plt.plot().
    :type kwargs: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    y = u4plotprep.get_forward_model(x, inv_results, direction)
    ax.plot(x, y, label="Forward Model", **kwargs)


@_add_or_create
def plot_fit_residuals(
    data: dict, fit_data: tuple, ax: Axes, direction: str = "UD", **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the residuals of the data and selected fit.

    :param x: The x axis (datetime).
    :type x: np.ndarray
    :param inv_results: A tuple of the inversion results.
    :type inv_results: tuple
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param direction: The direction of the inversion results. Options are: "EW",
        "NS", "UD", defaults to "UD"
    :type direction: str, optional
    :param kwargs: Additional arguments passed to plt.plot().
    :type kwargs: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """

    time = data["time"]
    y = np.nanmedian(data["timeseries"], axis=0)

    if len(fit_data) > 5:
        y_fit = u4plotprep.get_forward_model(time, fit_data, direction)

    y_res = y - y_fit

    ax.plot(time, y_res, ".", label="Residuals", **kwargs)


@_add_or_create
def add_tile(
    tiff_tile_path: os.PathLike,
    ax: Axes,
    vm: float = 0,
    imsize: int = 100,
    show: bool = True,
) -> Tuple[tuple, str]:
    """Adds a tiff file to the given axis.

    :param tiff_tile_path: The path to the tiff file.
    :type tiff_tile_path: os.PathLike
    :param ax: The axis to add the plot (optional).
    :type ax: Axes
    :param vm: Colormap minimum and maximum, defaults to 0
    :type vm: float, optional
    :param imsize: Resizes the image to this size, defaults to 100
    :type imsize: int, optional
    :param show: Adds the tile to the plot, defaults to True
    :type show: bool, optional
    :return: The boundaries and crs of the tile (bounds, crs).
    :rtype: Tuple[tuple, str]
    """
    tiff_tile = u4files.load_tiff(tiff_tile_path)
    if show:
        diff_tile = tiff_tile.read(1)
        if imsize:
            tile_resized = sktransf.resize(
                diff_tile, (imsize, imsize), anti_aliasing=True
            )
        else:
            tile_resized = diff_tile
        if not vm:
            vm = np.percentile(np.abs(tile_resized), 95)
        ax.imshow(
            tile_resized,
            cmap="RdYlBu",
            vmin=-vm,
            vmax=vm,
            extent=(
                tiff_tile.bounds.left,
                tiff_tile.bounds.right,
                tiff_tile.bounds.bottom,
                tiff_tile.bounds.top,
            ),
        )
    return tiff_tile.bounds, tiff_tile.crs
