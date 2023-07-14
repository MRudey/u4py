"""
Contains functions to modify or reformat data for plotting. This module helps to declutter the :func:`u4py.plotting.plots` or :func:`u4py.plotting.axes` modules.
"""

from typing import Callable, Tuple

import numpy as np
import scipy.optimize as spopt
import scipy.stats as spstats

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


def clean_points(
    points: np.ndarray, sigma: int = 2, fnc: Callable = spstats.norm
) -> np.ndarray:
    """Creates a numpy array for slicing that removes all outliers from the
    given points array. The outliers are defined by the standard deviation of
    the given statistical function which can be any scipy statistics function
    that supports standard deviations.

    :param points: The array of points to clean
    :type points: np.ndarray
    :param sigma: The amounts of standard deviations to use, defaults to 2
    :type sigma: int, optional
    :param fnc: The statistical function to use, defaults to spstats.norm
    :type fnc: Callable, optional
    :return: A numpy array that can be used to slice other arrays
    :rtype: np.ndarray
    """
    limit = sigma * fnc(*fnc.fit(points)).std()
    slc = np.nonzero(points > -limit) and np.nonzero(points < limit)
    return slc


def get_linfit_each_timeseries(data: dict) -> np.ndarray:
    """Fits each timeseries in the data dictionary. Suitable only for small
    datasets. Good for creating scatter plots.


    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :return: An array containing all linear fits in the dictionary.
    :rtype: np.ndarray
    """

    lintrend = []
    time = data["time"]
    for y in data["timeseries"]:
        slc = np.nonzero(np.isfinite(y))
        time_slice = time[slc]
        y = y[slc]
        time_days = np.linspace(0, len(time_slice) * 6, len(time_slice))

        lin_popt, _ = spopt.curve_fit(
            u4other.poly1,
            time_days,
            y,
        )
        lintrend.append(lin_popt[0] * 365)
    return lintrend


def _full_inv_single_comp(x: np.ndarray, *args) -> np.ndarray:
    """Returns a full fit including (semi-)annual sines and cosines

    :param x: The x axis.
    :type x: np.ndarray
    :param args: The fit arguments formatted as below.
    :type args: list
    :return: The forward model
    :rtype: np.ndarray

    Structure of *args:
    |    [0]: yaxis-offset
    |    [1]: linear trend
    |    [2]: semi-annual sine
    |    [3]: semi-annual cosine
    |    [4]: annual sine
    |    [5]: annual cosine
    """
    result = (
        args[0]  # yaxis-offset
        + args[1] * x  # linear trend
        + args[2] * np.sin(2 * np.pi * x)  # semi-annual sine
        + args[3] * np.cos(2 * np.pi * x)  # semi-annual cosine
        + args[4] * np.sin(4 * np.pi * x)  # annual sine
        + args[5] * np.cos(4 * np.pi * x)  # annual cosine
    )
    return result


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
    y = _full_inv_single_comp(time, *args)
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
