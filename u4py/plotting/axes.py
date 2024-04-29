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
from datetime import datetime
from functools import wraps
from typing import Callable, Iterable, Tuple

import contextily
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rioplot
import scipy.stats as spstats
import skimage.transform as sktransf
from matplotlib.axes import Axes
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from pyproj import CRS

import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.analysis.spatial as u4spatial
import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert


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
    ax.hist(y, "auto", density=True)
    xlims = ax.get_xlim()
    t_x = np.linspace(xlims[0], xlims[1], 100)
    stat_vals = fnc.fit(y)
    ax.plot(t_x, fnc.pdf(t_x, *stat_vals))


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
        label="95% Range",
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
    fit_num: int = 2,
    color: str = "C0",
    color_fit: str = "C1",
    annotate: bool = False,
    direction: str = "U",
    show_errors: bool = False,
) -> Tuple[Figure, Axes] | None:
    """Plots the fit data for a simple timeseries analysis.

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param results: A dictionary of loaded fit results, defaults to dict()
    :type results: dict, optional
    :param fit_num: Number of fit to plot (1=all points, 2=without outliers), defaults to 2 (without outliers)
    :type fit_num: int
    :param color: The color for the data, defaults to "C0"
    :type color: str, optional
    :param color_fit: The color for the fit, defaults to "C1"
    :type color_fit: str, optional
    :param annotate: Whether to add fit results as annotation to the plot
    :type annotate: bool, optional
    :param direction: The direction of results to use, defaults to "U": vertical
    :type direction: str, optional
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    if "signals" in results.keys():
        ax.plot(
            results["time"],
            results["signals"]["lin"] + results["signals"]["sin"],
            label="Simple Fit",
            color=color_fit,
        )
        # shift = time[0] - timedelta(days=sin_popt[2])

        # if shift.day > 10:
        #     prefix = "Mid"
        # elif shift.day > 20:
        #     prefix = "End"
        # else:
        #     prefix = "Start"
        if annotate:
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
                bbox={"boxstyle": "square", "fc": "white", "linewidth": 1},
            )
    else:
        quantiles = u4plotprep.get_timeseries_range(
            results[direction], results["t"]
        )
        plot_quantile_timeseries(
            quantiles["t_u"], quantiles, ax=ax, color=color
        )
        t_fly = u4convert.get_floatyear(quantiles["t_u"])
        t_q = np.linspace(np.min(t_fly), np.max(t_fly), 200)

        y_fit = u4plotprep._downsampled_forward_model(
            t_q,
            u4convert.get_floatyear(results[f"t_fit_{fit_num}"]),
            results[direction][f"y_fit_{fit_num}"],
        )
        ax.plot(
            u4convert.get_datetime(t_q),
            y_fit,
            color=color_fit,
            label="Inversion",
        )
        if show_errors:
            # The fit errors are only very roughly estimated from the residuals
            # Gets fit residuals
            yea = results["U"][f"y_fit_{fit_num}_err"]
            ys = np.std(yea)
            ym = np.median(yea)
            # Remove outliers in residuals
            ye = yea[yea > ym - ys]
            ye = ye[ye < ym + ys]
            fit_err = 2 * np.std(ye)
            ax.plot(
                u4convert.get_datetime(t_q),
                y_fit - fit_err,
                color=color_fit,
                linestyle=":",
                label="Fit error",
            )
            ax.plot(
                u4convert.get_datetime(t_q),
                y_fit + fit_err,
                color=color_fit,
                linestyle=":",
            )
        if annotate:
            res_name = ["inversion_results", "ori_inversion_results"]
            ax.annotate(
                u4invert.print_inversion_results(
                    results[direction][res_name[fit_num - 1]],
                    results[direction]["parameters_list"],
                ),
                (0.99, 0.01),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="bottom",
                bbox={"boxstyle": "square", "fc": "white", "linewidth": 1},
            )


@_add_or_create
def plot_region_trend(
    data_region: dict,
    ax: Axes,
    fit: bool = True,
    use_gdf: bool = True,
    vm: Tuple | float = 0,
    key: str = "",
    legend_args: dict = dict(),
) -> Tuple[Figure, Axes] | None:
    """Calculates and plots the regional trend in a region of data.

    :param data_region: Dictionary with multiple data dictionaries according to u4py standard (e.g. read from h5).
    :type data_region: dict
    :param ax: The axis to add the plot to (optional).
    :type ax: Axes
    :param fit: Whether to use linear fits or only the final value (optional), defaults to True.
    :type fit: bool
    :param use_gdf: Uses a geodataframe for the values instead of showing a grid (optional), defaults to True.
    :type use_gdf: bool
    :param vm: Minimum or maximum for plot colorscale (optional), defaults to 0. If is a Tuple uses the values as min and max, if a single value uses this as min and max, if 0 then use the 95% percentile of the absolute for min and max.
    :type vm: Tuple | float
    :param key: Key to use for plotting if not fitting (optional), defaults to "" which uses the final value of the timeseries.
    :type key: str
    :param legend_args: Arguments passed to gdf.plot()
    :type legend_args: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    if fit:
        region_trend = u4plotprep.get_linfit_each_timeseries(data_region)
    elif key:
        region_trend = data_region[key]
    else:
        region_trend = u4plotprep.get_final_each_timeseries(data_region)

    if not use_gdf:
        u4plotprep.matshow_region(data_region, region_trend, ax)
    else:
        gdf = u4spatial.xy_data_to_gdf(
            data_region["x"], data_region["y"], region_trend, crs="EPSG:32632"
        )
        if vm:
            if isinstance(vm, Iterable):
                vmin = vm[0]
                vmax = vm[1]
            else:
                vmin = -vm
                vmax = vm
        else:
            vmax = np.percentile(np.abs(region_trend), 95)
            vmin = -vmax
        leg_args = {
            "orientation": "vertical",
            "extend": "both",
            "label": "Mean Vertical Velocity (mm/a)",
        }
        if legend_args:
            leg_args.update(legend_args)
        gdf.plot(
            "data",
            ax=ax,
            cmap="RdYlBu",
            legend_kwds=leg_args,
            legend=True,
            markersize=3,
            vmax=vmax,
            vmin=vmin,
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
    :param crs: The target coordinate system as accepted by GeoPandas (default: {"EPSG:23032"}).
    :type crs: str
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
            if "name" in shape.keys():
                to_label = shape[shape["name"] == k]
            elif "station" in shape.keys():
                to_label = shape[shape["station"] == k]
            else:
                to_label = []
            if len(to_label) > 0:
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
    base_map_path: os.PathLike = None,
    ax: Axes = None,
    crs: str = "EPSG:23032",
    zoom: str | int = "auto",
    source=contextily.providers.OpenStreetMap.DE,
    **kwargs,
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the basemap as the lowest layer. If no basemap is given it is automatically loaded from osm.

    :param base_map_path:  Path to the geotiff with the basemap, defaults to None
    :type base_map_path: os.PathLike, optional
    :param ax: The axis to add the plot to (optional)., defaults to None
    :type ax: Axes, optional
    :param crs: The coordinate system of the axis, required for correct scaling of the basemap, defaults to "EPSG:23032".
    :type crs: str, optional
    :param zoom: The zoom level of the tiles. Ranges from 1 to 15, usually is about 10, defaults to "auto"
    :type zoom: str | int, optional
    :param kwargs:  Additional arguments passed to rasterio.plot.show(). See rasterio documentation.
    :type kwargs: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    if base_map_path:
        with rasterio.open(base_map_path) as base_map:
            rioplot.show(base_map, ax=ax, zorder=0, **kwargs)
    else:
        if source == contextily.providers.CartoDB.Voyager:
            contextily.add_basemap(
                ax=ax,
                crs=crs,
                source=contextily.providers.CartoDB.VoyagerNoLabels,
                zoom=zoom,
                zorder=0,
            )
            contextily.add_basemap(
                ax=ax,
                crs=crs,
                source=contextily.providers.CartoDB.VoyagerOnlyLabels,
                zoom=zoom,
                zorder=20,
            )
        else:
            contextily.add_basemap(
                ax=ax,
                crs=crs,
                source=source,
                zoom=zoom,
                zorder=0,
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
    data: dict,
    fit_data: tuple | dict,
    ax: Axes,
    direction: str = "UD",
    fit_num: int = 2,
    **kwargs,
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
    :param fit_num: Number of fit to plot (1=all points, 2=without outliers), defaults to 2 (without outliers)
    :type fit_num: int
    :param kwargs: Additional arguments passed to plt.plot().
    :type kwargs: dict
    :return: The figure and axis if there was no axis specified.
    :rtype: Tuple[Figure, Axes] | None
    """
    if isinstance(fit_data, tuple):
        time = data["time"]
        if isinstance(time[0], datetime):
            time_flt = u4convert.get_floatyear(time)
        y = np.nanmedian(data["timeseries"], axis=0)

        if len(fit_data) > 6:
            y_fit = u4plotprep.get_forward_model(time, fit_data, direction)
        else:
            y_fit = u4plotprep._downsampled_forward_model(time_flt, fit_data)

        y_res = y - y_fit

        ax.plot(time, y_res, ".", label="Residuals", **kwargs)

    elif isinstance(fit_data, dict):
        quantiles = u4plotprep.get_timeseries_range(
            time=fit_data[f"t_fit_{fit_num}"],
            y=fit_data["U"],
            key=f"y_fit_{fit_num}_err",
        )
        plot_quantile_timeseries(
            quantiles["t_u"], quantiles, ax=ax, color="C0"
        )


@_add_or_create
def add_tile(
    tiff_tile_path: os.PathLike,
    ax: Axes,
    vm: float | Iterable = 0,
    imsize: int = 0,
    show: bool = True,
    cmap: str = "RdYlBu",
    colorbar: dict = dict(),
    slope: bool = False,
    hillshade: bool = False,
    multidir: bool = False,
    **kwargs,
) -> Tuple[tuple, str]:
    """Adds a tiff file to the given axis.

    :param tiff_tile_path: The path to the tiff file.
    :type tiff_tile_path: os.PathLike
    :param ax: The axis to add the plot (optional).
    :type ax: Axes
    :param vm: Colormap minimum and maximum, defaults to 0. If none is given, then the value for vmin and vmax is determined as +- the 95 percentile of the absolute values. If an Iterable is given, then these are used as (vmin, vmax)
    :type vm: float | Iterable, optional
    :param imsize: Resizes the image to this size, defaults to 0
    :type imsize: int, optional
    :param show: Adds the tile to the plot, defaults to True
    :type show: bool, optional
    :param cmap: Colormap for the plot, defaults to "RdYlBu"
    :type cmap: str, optional
    :param colorbar: Arguments passed to the colorbar, defaults to empty dict()
    :type colorbar: dict, optional
    :param slope: Create a slope plot, defaults to False
    :type slope: bool, optional
    :param hillshade: Create a hillshade plot, defaults to False
    :type hillshade: bool, optional
    :param multidir: Whether to create a multidirectional hillshade, defaults to False
    :type multidir: bool, optional
    :param kwargs: Additional arguments passed to plt.plot().
    :type kwargs: dict
    :return: The boundaries and crs of the tile (bounds, crs).
    :rtype: Tuple[tuple, str]
    """
    with rasterio.open(tiff_tile_path) as tiff_tile:
        if show:
            diff_tile = tiff_tile.read(1)

            # Resizing image to smaller resolution
            if imsize:
                tile_resized = sktransf.resize(
                    diff_tile, (imsize, imsize), anti_aliasing=True
                )
            else:
                tile_resized = diff_tile

            # Minimum maximum for colorbar
            if not vm:
                vm = np.nanpercentile(np.abs(tile_resized), 95)
                vmin = -vm
                vmax = vm
            elif isinstance(vm, Iterable):
                vmin = vm[0]
                vmax = vm[1]
            elif isinstance(vm, (float, int)):
                vmin = -vm
                vmax = vm
            if not hillshade and not slope:
                ims = ax.imshow(
                    tile_resized,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=(
                        tiff_tile.bounds.left,
                        tiff_tile.bounds.right,
                        tiff_tile.bounds.bottom,
                        tiff_tile.bounds.top,
                    ),
                    **kwargs,
                )
            elif hillshade:
                if multidir:
                    ls = LightSource(azdeg=0, altdeg=45)
                    hs = ls.hillshade(tile_resized, vert_exag=5)
                    for azdeg in np.linspace(60, 300, 5):
                        ls = LightSource(azdeg=azdeg, altdeg=45)
                        hs += ls.hillshade(tile_resized, vert_exag=5)
                    hs /= 6
                else:
                    ls = LightSource(azdeg=315, altdeg=45)
                    hs = ls.hillshade(tile_resized, vert_exag=3)

                ims = ax.imshow(
                    hs,
                    cmap=cmap,
                    extent=(
                        tiff_tile.bounds.left,
                        tiff_tile.bounds.right,
                        tiff_tile.bounds.bottom,
                        tiff_tile.bounds.top,
                    ),
                    **kwargs,
                )
            elif slope:
                ims = ax.imshow(
                    u4spatial.dem_slope(tile_resized),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=(
                        tiff_tile.bounds.left,
                        tiff_tile.bounds.right,
                        tiff_tile.bounds.bottom,
                        tiff_tile.bounds.top,
                    ),
                    **kwargs,
                )
            ax.yaxis.set_inverted(False)
            if colorbar:
                plt.colorbar(ims, ax=ax, **colorbar)
        return tiff_tile.bounds, tiff_tile.crs
