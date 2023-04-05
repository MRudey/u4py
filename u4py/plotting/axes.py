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
import scipy.optimize as spopt
import scipy.stats as spstats
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyproj import CRS

import u4py.analysis.other as u4other
import u4py.plotting.preparation as u4plotprep


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
        ax -- The axis to add the plot to (default: {None}).
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
        ax -- The axis to add the plot to (default: {None}).

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
        ax -- The axis to add the plot to (default: {None}).

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
        ax -- The axis to add the plot to (default: {None}).

    Returns:
        The figure and axis if there was no axis specified.
    """

    # Data Preparation
    time = data["time"]
    y_med = np.nanmedian(data["timeseries"], axis=0)

    # Remove non finite elements
    slc = np.nonzero(np.isfinite(y_med))
    time = time[slc]
    y_med = y_med[slc]

    # Convert time to days
    time_days = np.linspace(0, (time[-1] - time[0]).days, len(time))

    # Linear component
    lin_popt, lin_pcov = spopt.curve_fit(
        u4other.poly1,
        time_days,
        y_med,
    )
    linear_component = u4other.poly1(time_days, *lin_popt)
    y_detrend = y_med - linear_component

    sin_popt, sin_pcov = spopt.curve_fit(
        u4other.sinefunc,
        time_days,
        y_detrend,
        p0=[2, 6 / 365.25, 0],
    )
    sinus_component = u4other.sinefunc(time_days, *sin_popt)
    # y_residual = y_detrend - sinus_component

    ax.plot(time, linear_component + sinus_component, label="Fit")
    # shift = time[0] - timedelta(days=sin_popt[2])

    # if shift.day > 10:
    #     prefix = "Mid"
    # elif shift.day > 20:
    #     prefix = "End"
    # else:
    #     prefix = "Start"
    ax.annotate(
        "Longterm Trend: %.1f mm/a" % (lin_popt[0] * 365)
        + "\nYearly Variation: $\\pm$%.1f mm" % (np.abs(sin_popt[0])),
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
        ax -- The axis to add the plot to (default: {None}).
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
