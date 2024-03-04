"""
Contains functions to modify or reformat data for plotting. This module helps to declutter the :func:`u4py.plotting.plots` or :func:`u4py.plotting.axes` modules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spinterp
import scipy.optimize as spopt
from matplotlib.axes import Axes
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.utils.convert as u4convert


def convert_results_for_grid(
    results: list, chunk_size: int = 250
) -> Tuple[list, list, list, list, int]:
    """Returns the loaded data prepared for gridding with :func:`make_gridded_data`.

    :param results: Inversion results as a list with `[xmid, ymid, components]`.
    :type results: list
    :param chunk_size: The chunksize used during data preparation for inversion, defaults to 250
    :type chunk_size: int, optional
    :return: A tuple of lists containing `[xmids, ymids, lintrend, sinusoid, chunk_size]`
    :rtype: Tuple[list, list, list, list, int]
    """

    ncomps = len(results[0][2])
    if ncomps < 5:  # Simple fit data
        converted_data = np.ones((4, len(results))) * np.nan
    else:  # Full inversion data
        converted_data = np.ones((7, len(results))) * np.nan

    for ii, (xmid, ymid, components) in enumerate(results):
        converted_data[0, ii] = xmid
        converted_data[1, ii] = ymid
        if components is not None:
            if ncomps < 5:  # Simple fit data
                converted_data[2, ii] = components[0] * 365.25
                converted_data[3, ii] = np.abs(components[1])
            else:  # Full inversion data
                for jj in range(5):
                    converted_data[2 + jj, ii] = components[jj + 13]

    calc_chunk_size = (
        np.round(
            np.mean(np.diff(np.sort(np.unique(converted_data[0, :])))) / 10
        )
        * 10
    )
    if np.abs(chunk_size - calc_chunk_size) > 100:
        raise UserWarning("The actual chunk size is off by more than 100 m!")
    return converted_data, chunk_size


def make_gridded_data(
    converted_data: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Converts the data given by :func:`convert_results_for_grid` into a nice gridded format for plotting with matplotlib's :func:`imshow`.

    :param converted_data: The data in a single 2D matrix.
    :type converted_data: np.ndarray
    :param chunk_size: Chunk size to generate X and Y Grid
    :type chunk_size: int
    :return: Linear and sinusoidal components each as array and the extend for plotting.
    :rtype: Tuple[np.ndarray, np.ndarray, tuple]
    """
    logging.info(f"Creating gridded data.")
    rows, cols = converted_data.shape
    minx = np.min(converted_data[0, :])
    maxx = np.max(converted_data[0, :]) + chunk_size
    miny = np.min(converted_data[1, :])
    maxy = np.max(converted_data[1, :]) + chunk_size
    x = np.arange(minx, maxx, chunk_size) + chunk_size
    y = np.arange(miny, maxy, chunk_size)
    extent = (
        np.min(x) - 0.5 * chunk_size,
        np.max(x) + 0.5 * chunk_size,
        np.min(y) - 0.5 * chunk_size,
        np.max(y) + 0.5 * chunk_size,
    )

    grids = np.ones((rows - 2, len(y), len(x))) * np.nan

    for col in range(cols):
        xn = int((converted_data[0, col] - minx) / chunk_size)
        yn = int((converted_data[1, col] - miny) / chunk_size)
        for ii in range(2, rows):
            grids[ii - 2, yn, xn] = converted_data[ii, col]

    return grids, extent


def get_linfit_each_timeseries(data: dict) -> np.ndarray:
    """Fits each timeseries in the data dictionary. Suitable only for small
    datasets. Good for creating scatter plots.


    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :return: An array containing all linear fits in the dictionary.
    :rtype: np.ndarray
    """

    lintrend = []
    lincov = []
    duration = data["time"][-1] - data["time"][0]
    time_year = np.linspace(
        0, duration.days / 365.25, len(data["timeseries"][0])
    )
    for y in tqdm(data["timeseries"], desc="Linear Fitting", leave=False):
        slc = np.nonzero(np.isfinite(y))
        y = y[slc]
        time = time_year[slc]

        lin_popt, lin_cov = spopt.curve_fit(u4other.poly1, time, y)
        lintrend.append(lin_popt[0])
        lincov.append(np.nanvar(y - u4other.poly1(time, *lin_popt)))
    return lintrend, lincov


def get_final_each_timeseries(data: dict) -> np.ndarray:
    """Gets the final deformation averaged over the whole time in the full timeseries.


    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :return: An array containing all last values in the dictionary.
    :rtype: np.ndarray
    """
    dt = data["time"][-1] - data["time"][0]
    dt_years = dt.days / 365
    vals = (
        np.array([y[np.isfinite(y)][-1] for y in data["timeseries"]])
        / dt_years
    )
    return vals


def _downsampled_forward_model(
    tq: np.ndarray, t: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Returns the forward model at the queried times.

    :param tq: The time points to query the data.
    :type tq: np.ndarray
    :param t: The time points for the fit data.
    :type t: np.ndarray
    :param results: The forward model data.
    :type results: np.ndarray
    :return: The forward model at the query points.
    :rtype: np.ndarray
    """
    uniq_t = np.unique(t)
    uniq_y = np.zeros_like(uniq_t)
    for ii, ut in enumerate(uniq_t):
        uniq_y[ii] = y[np.argwhere(ut == t)[0]]
    bspline = spinterp.make_interp_spline(uniq_t, uniq_y)
    return bspline(tq)


def get_forward_model(
    x: np.ndarray, inv_results: tuple, dir: str = "UD"
) -> np.ndarray:
    """Gets the forward model of the inversion results.

    :param x: The x axis (datetime).
    :type x: np.ndarray
    :param inv_results:  A tuple of the inversion results.
    :type inv_results: tuple
    :param dir: The direction of the inversion results. Options are: "EW",
        "NS", "UD", defaults to "UD"
    :type dir: str, optional
    :return: The forward model for the specified direction
    :rtype: np.ndarray
    """
    directions = {  # Index where to start looking for the right components
        "EW": 0,
        "NS": 6,
        "UD": 12,
    }
    ii = directions[dir]
    args = inv_results[ii : ii + 6]
    time = np.array([u4convert.get_floatyear(t) for t in x])
    y = _downsampled_forward_model(time, *args)
    return y


def get_timeseries_range(
    y: np.ndarray | dict, time: np.ndarray = None, key: str = "y"
) -> dict:
    """Takes a time array and a component and calculates the data ranges for a nice plot.

    :param y: The component array or dictionary
    :type y: dict
    :param time: The input time array, defaults to None
    :type time: np.ndarray
    :return: The data as [`t_unique`, `y_med`, `y_95`, `y_68`, `y_32`, `y_5`].
    :rtype: dict

    The input can be:

    - a multidimensional numpy array, e.g., a stack of stations
    - a dictionary from :func:`u4py.utils.convert.reformat_inversion_results`.

    The resulting dictionary contains only unique time steps and medians with quantiles at each of them.
    """
    if isinstance(y, np.ndarray):
        result = {
            "y_med": np.nanmedian(y, axis=0),
            "y_95": np.nanpercentile(y, q=95, axis=0),
            "y_68": np.nanpercentile(y, q=68, axis=0),
            "y_32": np.nanpercentile(y, q=32, axis=0),
            "y_5": np.nanpercentile(y, q=5, axis=0),
            "t_u": np.ndarray([]),
        }
    elif isinstance(y, dict) and time.any():
        t_u = np.unique(time)
        pre_mat = [y[key][time == t] for t in t_u]
        max_len = np.max([len(p) for p in pre_mat])
        for ii, p in enumerate(pre_mat):
            if len(p) < max_len:
                p_n = np.ones((max_len,)) * np.nan
                p_n[: len(p)] = p
                pre_mat[ii] = p_n
        result = get_timeseries_range(np.rot90(pre_mat))
        result["t_u"] = t_u

    return result


def matshow_region(data_region: dict, region_trend: np.ndarray, ax: Axes):
    """Plots the PSI data in a given region as a meshgrid.

    *Might cause distortion*

    :param data_region: The original data in the region.
    :type data_region: dict
    :param region_trend: The analyzed trend data from the region.
    :type region_trend: np.ndarray
    :param ax: The axis where to plot the data.
    :type ax: Axes
    """
    xx, yy = np.meshgrid(
        np.unique(data_region["x"]), np.unique(data_region["y"])
    )
    zz = np.ones_like(yy) * np.nan
    for x, y, z in zip(data_region["x"], data_region["y"], region_trend):
        zz[np.bitwise_and((xx == x), (yy == y))] = z

    rng = np.percentile(np.abs(region_trend), 95)
    sc = ax.imshow(
        zz,
        origin="lower",
        extent=(
            np.nanmin(data_region["x"]),
            np.nanmax(data_region["x"]),
            np.nanmin(data_region["y"]),
            np.nanmax(data_region["y"]),
        ),
        vmin=-rng,
        vmax=rng,
        cmap="RdYlBu",
        # label="PSI locations",
        zorder=1,
    )

    plt.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        extend="both",
        label="Mean Vertical Velocity (mm/a)",
    )


def print_inversion_results_for_publications(results: dict):
    """Prints the inversion results in a readable and in a table exportable way to the stdout.

    :param results: The reformatted results dictionary.
    :type results: dict
    """
    r2 = u4other.R_squared(results["U"]["y"], results["U"]["y_fit_2_err"])
    ad_r2 = u4other.adj_R_squared(
        r2, len(results["U"]["parameters_list"]), len(results["t"])
    )

    print(
        u4invert.print_inversion_results(
            results["U"]["inversion_results"],
            results["U"]["parameters_list"],
        )
    )
    print(f"Goodness of Fit (R²) = {r2:.3f} (adjusted={ad_r2:.3f})")
    a, peak = u4other.superpose(
        results["U"]["inversion_results"][2],
        -results["U"]["inversion_results"][3],
    )
    peak_time = (peak / (2 * np.pi)) * 365
    peak_date = datetime(2015, 1, 1, 0, 0, 0) + timedelta(days=peak_time)
    print(f"max. annual sines: +-{a:.2f}")
    print(f"annual peak time: {peak_date.day:02}.{peak_date.month:02}.")
    a2, peak2 = u4other.superpose(
        results["U"]["inversion_results"][4],
        -results["U"]["inversion_results"][5],
    )
    peak_time2 = (peak2 / (4 * np.pi)) * 365
    peak_date2 = datetime(2015, 1, 1, 0, 0, 0) + timedelta(days=peak_time2)
    peak_date2_alt = peak_date2 + timedelta(days=365 / 2)
    print(f"max. annual sines: +-{a2:.2f}")
    print(f"semi-annual peak time: {peak_date2.day:02}.{peak_date2.month:02}.")
    print(
        f"semi-annual peak time2: {peak_date2_alt.day:02}.{peak_date2_alt.month:02}."
    )

    val_0 = results["U"]["inversion_results"][1]
    val_1 = results["U"]["inversion_results"][2]
    val_2 = results["U"]["inversion_results"][3]
    val_3 = results["U"]["inversion_results"][4]
    val_4 = results["U"]["inversion_results"][5]

    print("------")
    print(f"{val_0:.2f}")
    print(f"{val_1:.2f}")
    print(f"{val_2:.2f}")
    print(f"{a:.2f}")
    print(f"{peak_date.day:02}.{peak_date.month:02}.")
    print(f"{val_3:.2f}")
    print(f"{val_4:.2f}")
    print(f"{a2:.2f}")
    print(
        f"{peak_date2.day:02}.{peak_date2.month:02}. and {peak_date2_alt.day:02}.{peak_date2_alt.month:02}."
    )

    print("------")
