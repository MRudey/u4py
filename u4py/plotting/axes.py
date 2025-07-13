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

import logging
import os
from datetime import datetime
from functools import wraps
from typing import Callable, Iterable, Tuple

import contextily
import geopandas as gp
import mapclassify  # Keep for user defined chloropleths
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyproj
import rasterio
import rasterio.plot as rioplot
import scipy.stats as spstats
import shapely as shp
import skimage.transform as sktransf
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, hsv_to_rgb
from matplotlib.figure import Figure
from shapely import plotting as shplt

import u4py.addons.web_services as u4web
import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.analysis.spatial as u4spatial
import u4py.io.files as u4files
import u4py.io.gpkg as u4gpkg
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert

OSM_AVAILABLE = True


def _add_or_create(internal_plot):
    """Decorator for all functions in this module

    Functions decorated can either add their plot to the axis (if `ax` is set).
    Otherwise the decorator creates a new plot with `plt.subplots`

    :param internal_plot: The function creating the plot.
    :type internal_plot: Callable
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
    shp_path: os.PathLike | str, ax: Axes, **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot from the shapefile in the given coordinate system.

    :param shp_path: The path to the shapefile containing the geometry.
    :type shp_path: os.PathLike | str
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
    elif isinstance(kwgs["crs"], pyproj.CRS):
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
    base_map_path: os.PathLike | str = None,
    ax: Axes = None,
    crs: pyproj.CRS | None = "EPSG:23032",
    zoom: str | int = "auto",
    source=contextily.providers.OpenStreetMap.Mapnik,
    attribution=None,
    **kwargs,
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the basemap as the lowest layer. If no basemap is given it is automatically loaded from osm.

    :param base_map_path:  Path to the geotiff with the basemap, defaults to None
    :type base_map_path: os.PathLike | str, optional
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
    global OSM_AVAILABLE
    if base_map_path:
        with rasterio.open(base_map_path) as base_map:
            rioplot.show(base_map, ax=ax, zorder=0, **kwargs)
    else:
        try:
            if OSM_AVAILABLE:
                if source == contextily.providers.CartoDB.Voyager:
                    contextily.add_basemap(
                        ax=ax,
                        crs=crs,
                        source=contextily.providers.CartoDB.VoyagerNoLabels,
                        zoom=zoom,
                        zorder=0,
                        attribution=attribution,
                    )
                    contextily.add_basemap(
                        ax=ax,
                        crs=crs,
                        source=contextily.providers.CartoDB.VoyagerOnlyLabels,
                        zoom=zoom,
                        zorder=20,
                        attribution=attribution,
                    )
                else:
                    contextily.add_basemap(
                        ax=ax,
                        crs=crs,
                        source=source,
                        zoom=zoom,
                        zorder=0,
                        attribution=attribution,
                    )
            else:
                logging.info(
                    "Connection to Server timed out. No Basemap Added."
                )
        except:
            logging.info("Connection to Server timed out. No Basemap Added.")
            OSM_AVAILABLE = False


@_add_or_create
def add_web_map_service(
    ax: Axes,
    crs: str,
    wms_url: str = "https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-maps.ows",
    wms_version: str = "1.3.0",
    layer: str = "he_dtk50",
):
    """Adds a WMS layer as basemap.

    :param ax: The axis to add the plot to.
    :type ax: Axes
    :param crs: The coordinate system of the axis.
    :type crs: str
    :param wms_url: The URL to the wms server, defaults to "https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-maps.ows"
    :type wms_url: str, optional
    :param wms_version: The version of the wms server, defaults to "1.3.0"
    :type wms_version: str, optional
    :param layer: The layer to use, defaults to "he_dtk50"
    :type layer: str, optional

    Currently, the HVBG Webserver does not deliver any valid data. Other WMS
    servers work.
    """
    # "Hack" to make axis scale correctly ??
    add_basemap(ax=ax, attribution=" ")

    bound_gdf = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )

    # Get resolution and resize to maximum size of tile for HVBG tiles.
    size = np.array(ax.figure.get_size_inches() * ax.figure.dpi, dtype=int)
    if np.any(size > 3000):
        size = (size / np.max(size)) * 3000

    # Get image data and show it
    img = u4web.wms_in_gdf_boundary(
        bound_gdf, wms_url, wms_version, layer, size
    )
    ax.imshow(
        img,
        extent=(
            bound_gdf.bounds.minx[0],
            bound_gdf.bounds.maxx[0],
            bound_gdf.bounds.miny[0],
            bound_gdf.bounds.maxy[0],
        ),
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
    tiff_tile_path: os.PathLike | str,
    ax: Axes,
    vm: float | Iterable = 0,
    imsize: int = 0,
    show: bool = True,
    cmap: str = "RdYlBu",
    colorbar: dict = dict(),
    **kwargs,
) -> Tuple[tuple, str]:
    """Adds a tiff file to the given axis.

    :param tiff_tile_path: The path to the tiff file.
    :type tiff_tile_path: os.PathLike | str
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

            ax.yaxis.set_inverted(False)
            if colorbar:
                cbax = ax.inset_axes((0.7, 0, 0.2, 1))
                cbax.axis("off")
                plt.colorbar(ims, ax=cbax, **colorbar)
                ax.set_position([0, 0, 0.85, 1])

        return tiff_tile.bounds, tiff_tile.crs


@_add_or_create
def add_diff_plan(
    region: gp.GeoDataFrame, tiff_folder: os.PathLike | str, ax: Axes
):
    """Adds a differential motion plot to the axis.

    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike | str
    :param ax: The axis to plot into.
    :type ax: Axes
    """
    coverage = u4tiff.get_tiff_coverage(tiff_folder)
    if region.crs != coverage.crs:
        coverage = coverage.to_crs(region.crs)
    geom = region.geometry
    part = coverage.clip(geom)
    if len(part) > 0:
        for ii, fpath in enumerate(part.path.to_list()):
            if ii == 0:
                add_tile(
                    fpath,
                    ax=ax,
                    vm=2,
                    cmap="RdBu",
                    colorbar={
                        "label": "Vertikaler Versatz (m)",
                        "shrink": 0.7,
                        "extend": "both",
                        # "pad": 0.05,
                    },
                )
            else:
                add_tile(fpath, ax=ax, vm=2, cmap="RdBu")
    else:
        logging.info("No diff plan tiles in region found.")


@_add_or_create
def add_dem(region: gp.GeoDataFrame, tiff_folder: os.PathLike | str, ax: Axes):
    """Adds a digital elevation model dataset to the axis.

    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike | str
    :param ax: The axis to plot into.
    :type ax: Axes
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(region, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    if len(file_list) > 0:
        for fpath in file_list:
            hs_fpath = u4tiff.get_terrain(fpath, terrain_feature="hillshade")
            add_tile(hs_fpath, ax=ax, cmap="bone")
    else:
        logging.info("No dem tiles in region found.")


@_add_or_create
def add_slope(
    region: gp.GeoDataFrame,
    tiff_folder: os.PathLike | str,
    ax: Axes,
    cmap: mcolors.Colormap = "inferno",
):
    """Adds a slope map to the axis.

    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike | str
    :param ax: The axis to plot into.
    :type ax: Axes
    :param cmap: The colormap to use for the plot.
    :type cmap: mcolors.Colormap
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(region, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    if len(file_list) > 0:
        for ii, fpath in enumerate(file_list):
            sl_fpath = u4tiff.get_terrain(fpath, terrain_feature="slope")
            if ii > 0:
                add_tile(sl_fpath, ax=ax, cmap=cmap, vm=(0, 60))
            else:
                add_tile(
                    sl_fpath,
                    ax=ax,
                    cmap=cmap,
                    vm=(0, 60),
                    colorbar={
                        "label": "Hangneigung (°)",
                        "shrink": 0.7,
                        "extend": "max",
                        # "pad": 0.05,
                    },
                )
    else:
        logging.info("No tiff tiles in region found.")


@_add_or_create
def add_aspect(
    region: gp.GeoDataFrame, tiff_folder: os.PathLike | str, ax: Axes
):
    """Adds an aspect map to the axis.

    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike | str
    :param ax: The axis to plot into.
    :type ax: Axes
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(region, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    if len(file_list) > 0:
        for ii, fpath in enumerate(file_list):
            as_fpath = u4tiff.get_terrain(fpath, terrain_feature="aspect")
            if ii > 0:
                add_tile(as_fpath, ax=ax, cmap="hsv", vm=(0, 360))
            else:
                add_tile(
                    as_fpath,
                    ax=ax,
                    cmap="hsv",
                    vm=(0, 360),
                    colorbar={
                        "label": "Exposition (°)",
                        "shrink": 0.7,
                        "ticks": [0, 90, 180, 270, 360],
                        "format": mticker.FixedFormatter(
                            ["N", "E", "S", "W", "N"]
                        ),
                        # "pad": 0.05,
                    },
                )
    else:
        logging.info("No tiff tiles in region found.")


@_add_or_create
def add_aspect_slope(
    region: gp.GeoDataFrame, tiff_folder: os.PathLike | str, ax: Axes
):
    """Adds an aspect-slope map to the axis.

    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike | str
    :param ax: The axis to plot into.
    :type ax: Axes
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(region, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    if len(file_list) > 0:
        for ii, fpath in enumerate(file_list):
            as_fpath = u4tiff.get_terrain(fpath, terrain_feature="aspect")
            sl_fpath = u4tiff.get_terrain(fpath, terrain_feature="slope")
            with rasterio.open(as_fpath) as tiff_tile:
                as_val = tiff_tile.read(1) / 360
                if ii == 0:
                    add_tile(
                        as_fpath,
                        ax=ax,
                        cmap="hsv",
                        vm=(0, 360),
                        colorbar={
                            "label": "Exposition (°)",
                            "shrink": 0.7,
                            "ticks": [0, 90, 180, 270, 360],
                            "format": mticker.FixedFormatter(
                                ["N", "E", "S", "W", "N"]
                            ),
                            # "pad": 0.05,
                        },
                    )
            with rasterio.open(sl_fpath) as tiff_tile:
                sl_val = tiff_tile.read(1) / 45
                sl_val[sl_val > 0.999] = 0.999
                rgb = hsv_to_rgb(
                    np.stack((as_val, sl_val, np.ones_like(sl_val)), axis=2)
                )
                ax.imshow(
                    rgb,
                    extent=(
                        tiff_tile.bounds.left,
                        tiff_tile.bounds.right,
                        tiff_tile.bounds.bottom,
                        tiff_tile.bounds.top,
                    ),
                )
    else:
        logging.info("No tiff tiles in region found.")


@_add_or_create
def add_gpkg_data_in_axis(
    gpkg_path: os.PathLike | str,
    table: str,
    ax: Axes,
    ax_crs: str | pyproj.CRS | None = "EPSG:32632",
    **plot_kwargs,
) -> mpatches.Patch:
    """Adds data from a gpkg file to the plot using the boundaries of the axis as the extend of the geometry

    :param gpkg_path: The path to the geodatabase with HLNUG data.
    :type gpkg_path: os.PathLike | str
    :param table: The sql table name to use.
    :type table: str
    :param ax: The axis to plot into.
    :type ax: Axes
    :param **plot_kwargs: Keyword arguments passed to the `GeoDataFrame.plot()` function.
    :type **plot_kwargs: dict
    :return: A legend handle for the data
    :rtype: mpatches.Patch
    """
    crs = u4sql.get_crs(gpkg_path, table)[0]
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=ax_crs
    ).to_crs(crs)
    data = u4gpkg.load_gpkg_data_region_ogr(
        region, gpkg_path, table, clip=False
    )
    if "column" in plot_kwargs.keys():
        if plot_kwargs["column"] not in data.keys():
            plot_kwargs["column"] = None
    # Special plotting for roads
    if "fclass" in plot_kwargs.keys() and isinstance(data, gp.GeoDataFrame):
        data = data[data["fclass"] == plot_kwargs["fclass"]]
        if not data.empty:
            if plot_kwargs["fclass"] == "motorway":
                # Orange buffered shape with black edges and black center line
                plot_kwargs["facecolor"] = "#ffb300"
                plot_kwargs["edgecolor"] = "k"
                plot_kwargs["linewidth"] = 0.5
                plot_kwargs["zorder"] = 4
                del plot_kwargs["fclass"]
                gdf = gp.GeoDataFrame(geometry=[data.buffer(25).unary_union])
                gdf.plot(ax=ax, **plot_kwargs)
                data.plot(ax=ax, color="k", linewidth=0.5)
                return mpatches.Patch(label="Autobahn", **plot_kwargs)
            elif plot_kwargs["fclass"] == "primary":
                # Yellow buffered shape with black edges
                plot_kwargs["facecolor"] = "#ffff00"
                plot_kwargs["edgecolor"] = "k"
                plot_kwargs["linewidth"] = 0.5
                plot_kwargs["zorder"] = 4
                del plot_kwargs["fclass"]
                gdf = gp.GeoDataFrame(geometry=[data.buffer(10).unary_union])
                gdf.plot(ax=ax, **plot_kwargs)
                return mpatches.Patch(label="Bundesstraße", **plot_kwargs)
            elif plot_kwargs["fclass"] == "secondary":
                # White buffered shape with black edges
                plot_kwargs["facecolor"] = "#ffffff"
                plot_kwargs["edgecolor"] = "k"
                plot_kwargs["linewidth"] = 0.5
                plot_kwargs["zorder"] = 4
                del plot_kwargs["fclass"]
                gdf = gp.GeoDataFrame(geometry=[data.buffer(10).unary_union])
                gdf.plot(ax=ax, **plot_kwargs)
                return mpatches.Patch(label="Landstraße", **plot_kwargs)
    else:
        if len(data) > 0:
            data.plot(ax=ax, **plot_kwargs)
        else:
            logging.info("No data inside axis found.")


@_add_or_create
def add_gpkg_data_where(
    gpkg_path: os.PathLike | str,
    table: str,
    where: str,
    ax: Axes,
    buffer: float = 0,
    **plot_kwargs,
):
    """Adds data from a gpkg file using a where clause, e.g., to only plot geometries in a certain group.

    :param gpkg_path: The path to the geodatabase.
    :type gpkg_path: os.PathLike | str
    :param table: The sql table to use.
    :type table: str
    :param where: The where clause used for filtering (in OGRSQL).
    :type where: str
    :param ax: The axis to plot into.
    :type ax: Axes
    :param buffer: A buffer distance used to improve the visibility of underlying features, defaults to 0.
    :type buffer: float
    :param **plot_kwargs: Keyword arguments passed to the `GeoDataFrame.plot()` function.
    :type **plot_kwargs: dict
    """
    data = u4gpkg.load_gpkg_data_where_ogr(gpkg_path, table, where)
    if len(data) > 0:
        if buffer:
            data.buffer(buffer).plot(ax=ax, **plot_kwargs)
        else:
            data.plot(ax=ax, **plot_kwargs)

    else:
        logging.info(f"No data found where: {where}")


@_add_or_create
def add_hlnug_shapes(
    geometries: list[shp.Polygon | shp.Point | shp.LineString],
    leg_dict: dict,
    ax: Axes,
):
    """Adds shapes loaded from a hlnug server to the axis using the symbology loaded from a legend file.

    :param geometries: A list of shapely geometries.
    :type geometries: list[shp.Polygon  |  shp.Point  |  shp.LineString]
    :param leg_dict: A dictionary with formatting options for each geometry in `geometries` with the same index.
    :type leg_dict: dict
    :param ax: The axis to add the data to.
    :type ax: Axes
    """
    for ii, poly in enumerate(geometries):
        if isinstance(leg_dict["hatch"][ii], str):
            shplt.plot_polygon(
                poly,
                ax=ax,
                fc=leg_dict["fc"][ii],
                ec=leg_dict["ec"][ii],
                fill=leg_dict["fill"][ii],
                hatch=leg_dict["hatch"][ii],
                alpha=leg_dict["alpha"][ii],
                linewidth=leg_dict["linewidth"][ii],
                add_points=False,
            )
        elif isinstance(leg_dict["hatch"][ii], dict):
            shplt.plot_polygon(
                poly,
                ax=ax,
                fc=leg_dict["fc"][ii],
                ec=leg_dict["ec"][ii],
                fill=leg_dict["fill"][ii],
                alpha=leg_dict["alpha"][ii],
                linewidth=leg_dict["linewidth"][ii],
                add_points=False,
            )
            for hii in range(len(leg_dict["hatch"][ii]["hatch"])):
                ht = leg_dict["hatch"][ii]["hatch"][hii]
                hc = leg_dict["hatch"][ii]["hatch_color"][0]
                hca = leg_dict["hatch"][ii]["hatch_alpha"][0]
                shplt.plot_polygon(
                    poly,
                    ax=ax,
                    fc="None",
                    ec=hc,
                    hatch=ht,
                    fill=leg_dict["fill"][ii],
                    alpha=hca,
                    linewidth=leg_dict["linewidth"][ii],
                    add_points=False,
                )


@_add_or_create
def add_fault_data(
    fault_data: gp.GeoDataFrame,
    leg_handles: list,
    leg_labels: list,
    ax: Axes,
) -> Tuple[list, list]:
    """Adds faults from the HLNUG server to the axis

    :param fault_data: The fault data loaded from HLNUG
    :type fault_data: gp.GeoDataFrame
    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param ax: The axis to add the data to
    :type ax: Axes
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    # Add known faults
    faults = ~fault_data.BEZEICHNUNG.str.contains(
        "vermutet", case=False, na=False
    )
    if faults.any():
        fault_data[faults].plot(ax=ax, color="k")
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Störung",
            color="k",
            linewidth=2,
            linestyle="-",
        )

    # Add suspected faults
    sus_faults = fault_data.BEZEICHNUNG.str.contains(
        "vermutet", case=False, na=False
    )
    if sus_faults.any():
        fault_data[sus_faults].plot(ax=ax, color="k", linestyle="--")
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Störung, vermutet",
            color="k",
            linewidth=2,
            linestyle="--",
        )
    return leg_handles, leg_labels


@_add_or_create
def add_hydrological_points(
    hydro_pts: gp.GeoDataFrame,
    leg_handles: list,
    leg_labels: list,
    ax: Axes,
) -> Tuple[list, list]:
    """Adds hydrogeological points of interest from the HLNUG server to the axis

    :param hydro_pts: The hydrogeological points of interest data loaded from HLNUG
    :type hydro_pts: gp.GeoDataFrame
    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param ax: The axis to add the data to
    :type ax: Axes
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    springs = hydro_pts.BEZEICHNUNG.str.contains("Quell", case=False, na=False)
    if springs.any():
        hydro_pts[springs].plot(
            ax=ax,
            marker=mpath.Path.arc(0, 180),
            color="#ff00cc",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Quellen und Quellgruppen",
            marker=mpath.Path.arc(0, 180),
            color="#ff00cc",
            linestyle="None",
        )
    ponors = hydro_pts.BEZEICHNUNG.str.contains(
        "Schwind", case=False, na=False
    )
    if ponors.any():
        hydro_pts[ponors].plot(
            ax=ax,
            marker="$\downarrow$",
            color="#ff00cc",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Schwinde, Bachschwinde",
            marker="$\downarrow$",
            color="#ff00cc",
            linestyle="None",
        )

    drains = hydro_pts.BEZEICHNUNG.str.contains("Drän", case=False, na=False)
    if drains.any():
        hydro_pts[drains].plot(
            ax=ax,
            marker="s",
            color="#ff00cc",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Drän",
            marker="s",
            color="#ff00cc",
            linestyle="None",
        )

    drain_adits = hydro_pts.BEZEICHNUNG.str.contains(
        "Wasserstollen", case=False, na=False
    )
    if drain_adits.any():
        hydro_pts[drain_adits].plot(
            ax=ax,
            marker="s",
            color="r",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Wasserstollen",
            marker="s",
            color="r",
            linestyle="None",
        )

    wells = hydro_pts.BEZEICHNUNG.str.contains("Brunnen", case=False, na=False)
    if wells.any():
        hydro_pts[wells].plot(
            ax=ax,
            marker="o",
            edgecolor="r",
            color="None",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Brunnen",
            marker="o",
            markeredgecolor="r",
            markerfacecolor="None",
            linestyle="None",
        )

    if not (springs | drains | ponors | drain_adits | wells).all():
        print("There are hydrogeological points without symbology.")
    return leg_handles, leg_labels


@_add_or_create
def add_soggy_areas(
    soggy_areas: gp.GeoDataFrame,
    leg_handles: list,
    leg_labels: list,
    ax: Axes,
) -> Tuple[list, list]:
    """Adds soggy areas from the HLNUG server to the axis

    :param soggy_areas: The soggy areas data loaded from HLNUG
    :type soggy_areas: gp.GeoDataFrame
    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param ax: The axis to add the data to
    :type ax: Axes
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    soggy_areas.plot(ax=ax, facecolor="b", edgecolor="None", alpha=0.5)
    add_patch_legend_entry(
        leg_handles,
        leg_labels,
        "Nassstellen",
        facecolor="b",
        edgecolor="None",
        alpha=0.5,
    )
    return leg_handles, leg_labels


@_add_or_create
def add_water_surface(
    water_surface: gp.GeoDataFrame,
    leg_handles: list,
    leg_labels: list,
    ax: Axes,
) -> Tuple[list, list]:
    """Adds water surfaces from the HLNUG server to the axis

    :param water_surface: The water surfaces data loaded from HLNUG
    :type water_surface: gp.GeoDataFrame
    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param ax: The axis to add the data to
    :type ax: Axes
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    water_surface.plot(ax=ax, facecolor="c", edgecolor="None")
    add_patch_legend_entry(
        leg_handles,
        leg_labels,
        "Wasserflächen",
        facecolor="c",
        edgecolor="None",
    )
    return leg_handles, leg_labels


@_add_or_create
def add_geo_pts(
    geo_pts: gp.GeoDataFrame,
    leg_handles: list,
    leg_labels: list,
    ax: Axes,
) -> Tuple[list, list]:
    """Adds geological points of interest from the HLNUG server to the axis

    :param geo_pts: The geological points of interest data loaded from HLNUG
    :type geo_pts: gp.GeoDataFrame
    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param ax: The axis to add the data to
    :type ax: Axes
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    landslides = geo_pts.BEZEICHNUNG.str.contains(
        "Rutschung", case=False, na=False
    )
    if landslides.any():
        geo_pts[landslides].plot(
            ax=ax,
            marker="$\downarrow$",
            edgecolor="r",
            facecolor="None",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Rutschung",
            marker="$\downarrow$",
            markeredgecolor="r",
            markerfacecolor="None",
            linestyle="None",
        )

    mining = geo_pts.BEZEICHNUNG.str.contains("Bergbau", case=False, na=False)
    if mining.any():
        geo_pts[mining].plot(
            ax=ax,
            marker="x",
            edgecolor="k",
            facecolor="None",
            markersize=50,
            zorder=5,
        )
        leg_handles, leg_labels = add_line_legend_entry(
            leg_handles,
            leg_labels,
            "Bergbau",
            marker="x",
            markeredgecolor="k",
            markerfacecolor="None",
            linestyle="None",
        )
    return leg_handles, leg_labels


@_add_or_create
def add_drill_sites(
    drill_sites: gp.GeoDataFrame,
    ax: Axes,
    leg_handles: list = [],
    leg_labels: list = [],
) -> Tuple[list, list]:
    """Adds drill sites from the HLNUG server to the axis

    :param drill_sites: The drill sites data loaded from HLNUG
    :type drill_sites: gp.GeoDataFrame
    :param ax: The axis to add the data to
    :type ax: Axes
    :param leg_handles: The legend handles for symbology, defaults to [].
    :type leg_handles: list
    :param leg_labels: The legend labels, defaults to [].
    :type leg_labels: list
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    drill_sites.plot(
        ax=ax,
        column="ET",
        scheme="UserDefined",
        classification_kwds={"bins": [10, 20, 50, 100]},
        cmap=ListedColormap(
            [
                [0.56078431, 0.76078431, 1.0, 1.0],
                [0.47058824, 0.61176471, 1.0, 1.0],
                [0.61960784, 0.50196078, 1.0, 1.0],
                [0.81176471, 0.34117647, 0.96862745, 1.0],
                [1.0, 0.0, 0.76078431, 1.0],
            ]
        ),
        edgecolor="k",
    )
    leg_handles, leg_labels = add_line_legend_entry(
        leg_handles, leg_labels, "Bohrungen", linestyle="None"
    )
    leg_handles, leg_labels = add_line_legend_entry(
        leg_handles,
        leg_labels,
        "< 10 m",
        marker="o",
        markerfacecolor=[0.56078431, 0.76078431, 1.0, 1.0],
        markeredgecolor="k",
        linestyle="None",
    )
    leg_handles, leg_labels = add_line_legend_entry(
        leg_handles,
        leg_labels,
        "10 - 20 m",
        marker="o",
        markerfacecolor=[0.47058824, 0.61176471, 1.0, 1.0],
        markeredgecolor="k",
        linestyle="None",
    )
    leg_handles, leg_labels = add_line_legend_entry(
        leg_handles,
        leg_labels,
        "20 - 50 m",
        marker="o",
        markerfacecolor=[0.61960784, 0.50196078, 1.0, 1.0],
        markeredgecolor="k",
        linestyle="None",
    )
    leg_handles, leg_labels = add_line_legend_entry(
        leg_handles,
        leg_labels,
        "50 - 100 m",
        marker="o",
        markerfacecolor=[0.81176471, 0.34117647, 0.96862745, 1.0],
        markeredgecolor="k",
        linestyle="None",
    )
    leg_handles, leg_labels = add_line_legend_entry(
        leg_handles,
        leg_labels,
        "> 100 m",
        marker="o",
        markerfacecolor=[1.0, 0.0, 0.76078431, 1.0],
        markeredgecolor="k",
        linestyle="None",
    )
    return leg_handles, leg_labels


def add_line_legend_entry(
    leg_handles: list, leg_labels: list, label: str, **kwargs
) -> Tuple[list, list]:
    """Helper function to add handles and labels of a line object to the list.

    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param **kwargs: Formatting arguments for the Line2D object
    :type **kwargs: dict
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    leg_handles.append(mlines.Line2D([], [], **kwargs))
    leg_labels.append(label)
    return leg_handles, leg_labels


def add_patch_legend_entry(
    leg_handles: list, leg_labels: list, label: str, **kwargs
) -> Tuple[list, list]:
    """Helper function to add handles and labels of a patch object to the list.

    :param leg_handles: The legend handles for symbology
    :type leg_handles: list
    :param leg_labels: The legend labels
    :type leg_labels: list
    :param **kwargs: Formatting arguments for the Line2D object
    :type **kwargs: dict
    :return: A tuple with the updated legend handles and labels
    :rtype: Tuple[list, list]
    """
    leg_handles.append(mpatches.Patch(**kwargs))
    leg_labels.append(label)
    return leg_handles, leg_labels
