"""
Contains functions to create plots from a data dictionary. All functions
should follow the following template:

    plot_FUNCNAME(data:dict, ax: Axes=None, KEYWORDS) -> None, Tuple[Figure, Axes]

- Functions may also be used to add the plot to an existing plot. If given an
  `Axes` the function will plot into the given axis (useful for subplots). If
  no `Axes` is given the function will create a Figure and return both, an
  `Axes` and `Figure` object.
- The input `data` is required.
- For flexibility keywords can be added.
"""

import os
from typing import Callable, Tuple

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rioplot
import scipy.optimize as spopt
import scipy.stats as spstats
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyproj import CRS

import u4py.analysis.other as u4other
import u4py.analysis.processing as u4proc
import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert


def add_or_create(internal_plot):
    """Decorator for all functions in this module

    Functions decorated can either add their plot to the axis (if `ax` is set).
    Otherwise the decorator creates a new plot with `plt.subplots`

    """

    def wrapper_internal_plot(*args, ax=None, **kwargs):
        # Plotting
        if not ax:
            fig, ax = plt.subplots()
            internal_plot(*args, ax=ax, **kwargs)
            return fig, ax
        else:
            internal_plot(*args, ax=ax, **kwargs)

    return wrapper_internal_plot


@add_or_create
def plot_stat_func(
    data: dict, ax: Axes, stat_fnc: Callable = np.nanmean
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the given statistic.

    Arguments:
        data -- Data dictionary according to u4py standard (e.g. read from h5).

    Keyword Arguments:
        ax -- The axis to add the plot to.
        stat_fnc -- The statistical function to use on the timeseries data. (default: {np.nanmean})

    Returns:
        The figure and axis if there was no axis specified.
    """

    time = data["time"]
    y = stat_fnc(data["timeseries"], axis=0)
    ax.plot(time, y)


@add_or_create
def plot_cwt(data: dict, ax: Axes) -> Tuple[Figure, Axes] | None:
    """Does a continous wavelet transform of the data to find dominant frequencies and plots it.

    Arguments:
        data -- Data dictionary according to u4py standard (e.g. read from h5).

    Keyword Arguments:
        ax -- The axis to add the plot to.

    Returns:
        The figure and axis if there was no axis specified.
    """

    time = data["time"]
    y = np.median(data["timeseries"], axis=0)
    freqs, coi, power = u4other.cwt(y, 6)
    xx, yy = np.meshgrid(time, freqs)
    ax.contourf(xx, yy, power, levels=20)
    ax.fill_between(time, coi, np.min(freqs), color="w", alpha=0.25)


@add_or_create
def plot_pdf(
    y: np.ndarray, ax: Axes, fnc: spstats.rv_continuous = spstats.norm
) -> (
    spstats.distributions.rv_frozen
    | Tuple[spstats.distributions.rv_frozen, Figure, Axes]
):
    """Adds a plot of the probability density function.

    Arguments:
        y -- The data to be used to fit the distribution


    Keyword Arguments:
        ax -- The axis to add the plot to (default: {None})
        fnc -- Distribution class used to fit the data (default: {spstats.norm})

    Returns:
        Returns the frozen distribution object or a tuple with object, figure and axis.
    """
    raise NotImplementedError("Implement the return of stat vals...")
    # xlims = ax.get_xlim()
    # ax.hist(y, "auto", density=True)
    # t_x = np.linspace(xlims[0], xlims[1], 100)
    # stat_vals = fnc.fit(y)
    # ax.plot(t_x, fnc.pdf(t_x, *stat_vals))


@add_or_create
def plot_timeseries(
    x: np.ndarray, y: np.ndarray, ax: Axes, color: str = "C0"
) -> Tuple[Figure, Axes] | None:
    """Creates a nice plot of a timeseries including the range of values.

    Arguments:
        x -- 1D Array containing the data for the x-axis.
        y -- 1D or 2D Array containing the data for the y-axis.

    Keyword Arguments:
        ax -- The axis to add the plot to.

    Returns:
        The figure and axis if there was no axis specified.
    """

    y_med = np.nanmedian(y, axis=0)
    y_95 = np.nanpercentile(y, q=95, axis=0)
    y_68 = np.nanpercentile(y, q=68, axis=0)
    y_32 = np.nanpercentile(y, q=32, axis=0)
    y_5 = np.nanpercentile(y, q=5, axis=0)
    ax.plot(x, y_med, ".", label="Median")
    ax.fill_between(
        x,
        y_95,
        y_5,
        color=color,
        alpha=0.5,
        edgecolor=None,
        label="Data Range",
    )
    ax.fill_between(x, y_68, y_32, color=color, alpha=0.5, edgecolor=None)


@add_or_create
def plot_timeseries_fit(data: dict, ax: Axes) -> Tuple[Figure, Axes] | None:
    """Plots the fit data for a simple timeseries analysis.

    Arguments:
        data -- Data dictionary according to u4py standard (e.g. read from h5).

    Keyword Arguments:
        ax -- The axis to add the plot to.

    Returns:
        The figure and axis if there was no axis specified.
    """

    results = u4proc.get_decomposed_signals(data)

    ax.plot(
        results["time"],
        results["signals"]["lin"] + results["signals"]["sin"],
        label="Fit",
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


@add_or_create
def plot_region_trend(data_region: dict, ax: Axes):
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


@add_or_create
def add_shapefile(
    shp_path: os.PathLike, ax: Axes, **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot from the shapefile in the given coordinate system.

    Arguments:
        shp_path -- The path to the shapefile containing the geometry.

    Keyword Arguments:
        ax -- The axis to add the plot to.
        crs --  The target coordinate system as accepted by GeoPandas (default: {"EPSG:23032"}).
        kwargs -- Additional arguments passed to shape.plot(). See GeoPandas documentation.
    Returns:
        The figure and axis if there was no axis specified.
    """
    kwgs = {"crs": "EPSG:23032"}
    kwgs.update(kwargs)
    shape = gp.read_file(shp_path)
    if isinstance(kwgs["crs"], str):
        if shape.crs.to_string() != kwgs["crs"]:
            shape = shape.to_crs(kwgs["crs"])
    elif isinstance(kwgs["crs"], CRS):
        shape = shape.to_crs(kwgs["crs"])
    kwgs.pop("crs")
    shape.plot(ax=ax, **kwgs)


@add_or_create
def add_basemap(
    base_map_path: os.PathLike, ax: Axes, **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the basemap as the lowest layer

    Arguments:
        base_map_path -- Path to the geotiff with the basemap
        ax --  The axis to add the plot to.
        kwargs -- Additional arguments passed to rasterio.plot.show(). See rasterio documentation.

    Returns:
        The figure and axis if there was no axis specified.
    """
    with rasterio.open(base_map_path) as base_map:
        rioplot.show(base_map, ax=ax, zorder=0, **kwargs)


@add_or_create
def plot_inversion_fit(
    x: np.ndarray,
    inv_results: tuple,
    ax: Axes,
    direction: str = "UD",
    **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot from inversion results.

    Arguments:
        x -- The x axis (datetime).
        inv_results -- A tuple of the inversion results.
        ax -- The axis to add the plot to.

    Keyword Arguments:
        direction -- The direction of the inversion results. Options are: "EW",
        "NS", "UD" (default {"UD"})
        kwargs -- Additional arguments passed to plt.plot().

    Returns:
        The figure and axis if there was no axis specified.
    """
    y = u4plotprep.get_forward_model(x, inv_results, direction)
    ax.plot(x, y, label="Forward Model", **kwargs)


@add_or_create
def plot_fit_residuals(
    data: dict, fit_data: tuple, ax: Axes, direction: str = "UD", **kwargs
) -> Tuple[Figure, Axes] | None:
    """Creates a plot with the residuals of the data and selected fit

    Arguments:
        data -- The data dictionary.
        fit_data -- A tuple of fit data, the length defines the type
        ax -- The axis to add the plot to.

    Keyword Arguments:
        direction -- The direction of the inversion results. Options are: "EW",
        "NS", "UD" (default {"UD"})
        kwargs -- Additional arguments passed to plt.plot().

    Returns:
        The figure and axis if there was no axis specified.
    """

    time = data["time"]
    y = np.nanmedian(data["timeseries"], axis=0)

    if len(fit_data) > 5:
        y_fit = u4plotprep.get_forward_model(time, fit_data, direction)

    y_res = y - y_fit

    ax.plot(time, y_res, ".", label="Residuals", **kwargs)
