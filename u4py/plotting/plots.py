"""
Contains functions with ready made plots. This module uses axis functions defined in :func:`u4py.plotting.axes` to create more complicated plots. The functions also do some processing and other data modification. Each function should contain a `save_path` if possible where the output figure is saved. If none is given the figure is shown interactively, otherwise it is saved and then destroyed.
"""
import os
from datetime import datetime, timedelta

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import scipy.stats as spstats
import uncertainties as unc

import u4py.analysis.other as u4other
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.plotting.preparation as u4plotprep


def plot_inversion_results(
    time: np.ndarray,
    time2: np.ndarray = np.array([]),
    data: dict = dict(),
    inversion_results: dict = dict(),
    save_path: os.PathLike = None,
):
    """Plots the results of a full inversion.

    :param time: The time axis of the first fit.
    :type time: np.ndarray
    :param time2: The time axis of the second fit (without outliers).
    :type time2: np.ndarray
    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param inversion_results: The results formatted as a dictionary.
    :type inversion_results: dict
    :param save_path: The path where to save the figure, defaults to None
    :type save_path: os.PathLike, optional
    """
    ew_mov = inversion_results["matrix_ori"][1]
    ud_mov = inversion_results["matrix_ori"][13]
    ux = np.unique(time[0])

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)
    y = [np.nanmedian(data["dataE"][np.argwhere(time[0] == uu)]) for uu in ux]
    yep = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 95)
        for uu in ux
    ]
    yed = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 5)
        for uu in ux
    ]
    yep2 = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 68)
        for uu in ux
    ]
    yed2 = [
        np.nanpercentile(data["dataE"][np.argwhere(time[0] == uu)], 32)
        for uu in ux
    ]
    axes[0].set_title("East-West Component")
    axes[0].plot(ux, y, ".", label="Median")
    axes[0].fill_between(
        ux, yed, yep, color="C0", alpha=0.5, edgecolor=None, label="95% range"
    )
    axes[0].fill_between(
        ux,
        yed2,
        yep2,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        label="68% range",
    )
    axes[0].plot(
        time[0], data["ori_dhat_data"]["dhatE"], color="C1", label="Fit"
    )
    if time2.size > 0:
        axes[0].plot(
            time2[0],
            data["dhat_data"]["dhatE"],
            color="C2",
            label="Fit (w/o outliers)",
        )
    axes[0].annotate(
        f"{ew_mov:.2} mm/yr", (0.05, 0.05), xycoords="axes fraction"
    )
    axes[0].legend(loc="best", fontsize="small", markerscale=0.5)
    # axes[1][0].plot(time[0], data["ori_dhat_data"]["dhatE"], "C1")
    y = [np.nanmedian(data["dataU"][np.argwhere(time[0] == uu)]) for uu in ux]
    yep = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 95)
        for uu in ux
    ]
    yed = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 5)
        for uu in ux
    ]
    yep2 = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 68)
        for uu in ux
    ]
    yed2 = [
        np.nanpercentile(data["dataU"][np.argwhere(time[0] == uu)], 32)
        for uu in ux
    ]
    axes[1].set_title("Vertical Component")
    axes[1].plot(ux, y, ".")
    axes[1].fill_between(ux, yed, yep, color="C0", alpha=0.5, edgecolor=None)
    axes[1].fill_between(ux, yed2, yep2, color="C0", alpha=0.5, edgecolor=None)
    axes[1].plot(time[0], data["ori_dhat_data"]["dhatU"])
    if time2.size > 0:
        axes[1].plot(time2[0], data["dhat_data"]["dhatU"])
    axes[1].annotate(
        f"Hebung/Senkung = {ud_mov:.2} mm/yr",
        (0.05, 0.05),
        xycoords="axes fraction",
    )
    # axes[1][1].plot(time[0], data["ori_dhat_data"]["dhatU"], "C1")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_iterative_fit(data: dict, save_path: os.PathLike = None):
    """Creates a plot with step by step fitting each in a separate subplot.

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param save_path: The path where to save the figure, defaults to None
    :type save_path: os.PathLike, optional
    """
    # Data preparation
    time = data["time"]
    y = np.nanmedian(data["timeseries"], axis=0)

    # Remove non finite elements
    slc = np.nonzero(np.isfinite(y))
    time = time[slc]
    y = y[slc]

    # Convert time to days
    time_days = np.linspace(0, len(time) * 6, len(time))

    # Linear component
    lin_popt, lin_pcov = spopt.curve_fit(
        u4other.poly1,
        time_days,
        y,
    )
    linear_component = u4other.poly1(time_days, *lin_popt)
    y_detrend = y - linear_component

    # Seasonal component
    popt, pcov = spopt.curve_fit(
        u4other.cosinefunc,
        time_days,
        y_detrend,
        p0=[2, 6 / 365.25, 0],
    )
    sinus_component = u4other.cosinefunc(time_days, *popt)
    y_residual = y_detrend - sinus_component

    # Create plot
    fig, axes = plt.subplots(
        ncols=3, figsize=(15, 5), sharex=True, sharey=True
    )
    axes[0].set_title("Linear Trend")
    # axes[0].plot(time, y, ".")
    axes[0].plot(time, y, ".-", linewidth=0.5)
    axes[0].plot(time, linear_component)
    axes[0].annotate(
        "Jährliche Hebung/Senkung: %.1f mm/a" % ((lin_popt[0]) * 365.25),
        (0.95, 0.05),
        xycoords="axes fraction",
        horizontalalignment="right",
    )

    axes[1].set_title("Signal - Linear = Sinusoidal Trend")
    # axes[1].plot(time, y_detrend, ".")
    axes[1].plot(time, y_detrend, ".-", linewidth=0.5)
    axes[1].plot(time, sinus_component)
    shift = datetime(2015, 1, 1) - timedelta(days=np.abs(popt[2]))

    if shift.day > 10:
        prefix = "Mitte"
    elif shift.day > 20:
        prefix = "Ende"
    else:
        prefix = "Anfang"

    axes[1].annotate(
        "Jährliche Schwankung: $\\pm$%.1f mm\nPeriodizität: %i Tage\nMaximum: %s %s"
        % (np.abs(popt[0]), (1 / popt[1]) * 6, prefix, shift.strftime("%B")),
        (0.95, 0.95),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="top",
    )

    axes[2].set_title("Residuals")
    # axes[2].plot(time, y_residual, ".")
    axes[2].plot(time, y_residual, ".-", linewidth=0.5)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_statistics(
    data: dict, timeslot: slice, save_path: os.PathLike = None
):
    """Plots some statistical characteristics of the data, e.g., PSI density.

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param timeslot: Slice with a time window.
    :type timeslot: slice
    :param save_path: The path where to save the figure, defaults to None
    :type save_path: os.PathLike, optional
    """
    yerr_min = np.percentile(data["timeseries"], 5, axis=0)
    yerr_max = np.percentile(data["timeseries"], 95, axis=0)
    rng = np.max(np.abs([np.min(yerr_min), np.max(yerr_max)]))
    points = data["timeseries"][:, timeslot]

    slc = u4plotprep.clean_points(points, fnc=spstats.t)

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes[0].set_title("PSI - Map")
    axes[0].scatter(
        data["x"][slc],
        data["y"][slc],
        c=points[slc],
        marker=".",
        cmap="RdBu",
        vmin=-rng,
        vmax=rng,
    )

    axes[1].set_title("PSI Density")
    axes[1].hexbin(data["x"], data["y"], gridsize=10)
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])

    axes[2].set_title("All PSI - Single Interval")
    axes[2].hist(data["timeseries"][:, timeslot], "auto", density=True)

    stat_dist = u4ax.plot_pdf(
        data["timeseries"][:, timeslot], ax=axes[2], fnc=spstats.t
    )
    u4ax.plot_pdf(
        data["timeseries"][:, timeslot], ax=axes[2], fnc=spstats.norm
    )

    mean_val = unc.ufloat(stat_dist.median(), 2 * stat_dist.std())
    axes[2].annotate(
        "N = %i\nMedian = %s mm"
        % (len(data["timeseries"][:, timeslot]), mean_val),
        (0.05, 0.95),
        xycoords="axes fraction",
        verticalalignment="top",
    )

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_gridded(
    lin_2d: np.ndarray,
    sin_2d: np.ndarray,
    extent: tuple,
    suptitle: str = "",
    base_map_path: os.PathLike = None,
    tektonik_path: os.PathLike = None,
    roi: gp.GeoDataFrame = None,
    save_path: os.PathLike = None,
    perc: int = 95,
    dpi: int = 300,
):
    """Creates a plot for gridded data

    :param lin_2d: A 2D Array containing the linear trend data.
    :type lin_2d: np.ndarray
    :param sin_2d: A 2D Array containing the seasonal variation data.
    :type sin_2d: np.ndarray
    :param extent: The extend of the 2D grid as (minx, maxx, miny, maxy) tuple.
    :type extent: tuple
    :param suptitle: The title for the plot, defaults to ""
    :type suptitle: str, optional
    :param base_map_path: Path to the basemap, defaults to None
    :type base_map_path: os.PathLike, optional
    :param tektonik_path: Path to the shape file with tectonic information, defaults to None
    :type tektonik_path: os.PathLike, optional
    :param roi: GeoDataFrame containing the regions of interest for detailed plots, defaults to None
    :type roi: gp.GeoDataFrame, optional
    :param save_path: Path where to save the plot, defaults to None
    :type save_path: os.PathLike, optional
    :param perc:  Percentile for the visualization, defaults to 95
    :type perc: int, optional
    :param dpi: Resolution of the plot for saving to png, defaults to 300
    :type dpi: int, optional
    """
    # Size of Figure (adapted to region of interest)
    figwidth = 11.7
    figheight = 8.27
    bounds = ()
    if roi is not None:
        bounds = roi.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        ratio = width / height
        figwidth = ratio * 1.25 * figwidth

    fig, axes = plt.subplots(
        ncols=2,
        sharex=True,
        sharey=True,
        dpi=dpi,
        figsize=(figwidth, figheight),
        layout="constrained",
    )

    # Add linear component
    rng = np.nanpercentile(np.abs(lin_2d), perc)
    linplt = axes[0].imshow(
        lin_2d,
        vmin=-rng,
        vmax=rng,
        origin="lower",
        extent=extent,
        cmap="turbo",
        zorder=1,
        alpha=0.8,
    )
    plt.colorbar(
        linplt,
        ax=axes[0],
        label="Displacement (mm/a)",
        orientation="horizontal",
        extend="both",
        shrink=0.5,
    )

    # Add seasonal component
    rng = np.nanpercentile(sin_2d, perc)
    sinplt = axes[1].imshow(
        sin_2d,
        vmin=0,
        vmax=rng,
        origin="lower",
        extent=extent,
        zorder=1,
        alpha=0.8,
    )
    plt.colorbar(
        sinplt,
        ax=axes[1],
        label="Amplitude (mm)",
        orientation="horizontal",
        extend="both",
        shrink=0.5,
    )

    # Fromatting and basemaps
    for ax in axes:
        if base_map_path:
            u4ax.add_basemap(base_map_path, ax=ax)
            u4plotfmt.add_copyright("Basemap: OSM", ax=ax)
        if tektonik_path:
            u4ax.add_shapefile(
                tektonik_path, ax=ax, color="k", zorder=2, linewidth=1
            )
        if roi is not None:
            ax.plot(*roi.exterior.xy, color="k")

    axes[0].set_title("Linear Component", fontweight="bold")
    axes[1].set_title("Seasonal Component", fontweight="bold")
    if suptitle:
        fig.suptitle(suptitle, fontsize="large", fontweight="bold")
    if bounds:
        axes[0].set_xlim(bounds[0], bounds[2])
        axes[0].set_ylim(bounds[1], bounds[3])
    u4plotfmt.map_style(ax=axes[0])
    u4plotfmt.map_style(ax=axes[1])
    # Save or show plot
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
