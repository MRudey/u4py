"""
Contains functions with ready made plots. Each function should contain a
save_path if possible.
"""
import os
from datetime import datetime, timedelta
from typing import Callable, Tuple

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot as rioplot
import scipy.ndimage as spimg
import scipy.optimize as spopt
import scipy.stats as spstats
import uncertainties as unc
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import u4py.analysis.other as u4other
import u4py.analysis.processing as u4process
import u4py.plotting.axes as u4ax
import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files


def plot_inversion_results(
    time, time2, data, inversion_results, save_path=None
):
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
    else:
        plt.show()


def plot_iterative_fit(data: dict, save_path: os.PathLike = None):
    """Creates a plot with step by step fitting each in a separate subplot.

    Arguments:
        data -- Data dictionary according to u4py standard (e.g. read from h5).

    Keyword Arguments:
        save_path -- The Path where to store the data (default: {None})
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
        u4other.sinefunc,
        time_days,
        y_detrend,
        p0=[2, 6 / 365.25, 0],
    )
    sinus_component = u4other.sinefunc(time_days, *popt)
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

    return fig, axes


def plot_statistics(data, timeslot):
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

    plt.show()
