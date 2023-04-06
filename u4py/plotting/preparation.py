""" Contains functions to modify or reformat data for plotting """

from typing import Callable, Tuple

import numpy as np
import scipy.ndimage as spimg
import scipy.optimize as spopt
import scipy.stats as spstats

import u4py.analysis.other as u4other
import u4py.utils.convert as u4convert


def convert_results_for_grid(
    results: list, chunk_size: int = 250
) -> Tuple[list, list, list, list, int]:
    """Returns the loaded data prepared for gridding"""
    xmids = []
    ymids = []
    lintrend = []
    sinusoid = []
    for xmid, ymid, components in results:
        xmids.append(xmid)
        ymids.append(ymid)
        if components is not None:
            if len(components) < 5:  # Simple fit data
                lintrend.append(components[0] * 365.25)
                sinusoid.append(np.abs(components[1]))
            else:  # Full inversion data
                lintrend.append(components[13])
                sinusoid.append(np.abs(components[14]))
        else:
            sinusoid.append(np.nan)
    return xmids, ymids, lintrend, sinusoid, chunk_size


def make_gridded_data(
    xmids: np.ndarray,
    ymids: np.ndarray,
    lintrend: np.ndarray,
    sinusoid: np.ndarray,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Converts the data into a nice gridded format for plotting with mpl.

    Arguments:
        xmids -- The x coordinates of the midpoints.
        ymids -- The y coordinates of the midpoints.
        lintrend -- The linear trend for the region
        sinusoid -- The sinusoidal variation for the region
        chunk_size -- Chunk size to generate X and Y Grid

    Returns:
        Linear and sinusoidal components each as array and the extend for plotting.
    """
    minx = np.min(xmids)
    maxx = np.max(xmids) + chunk_size
    miny = np.min(ymids)
    maxy = np.max(ymids) + chunk_size
    extent = (minx, maxx, miny, maxy)
    x = np.arange(minx, maxx, chunk_size)
    y = np.arange(miny, maxy, chunk_size)

    XX, YY = np.meshgrid(x, y)
    lin_2d = np.ones_like(XX) * np.nan
    sin_2d = np.ones_like(XX) * np.nan

    for xi, yi, li, si in zip(xmids, ymids, lintrend, sinusoid):
        xn = int((xi - minx) / chunk_size)
        yn = int((yi - miny) / chunk_size)
        lin_2d[yn, xn] = li
        sin_2d[yn, xn] = si

    lin_2d = spimg.median_filter(lin_2d, 3)
    sin_2d = spimg.median_filter(sin_2d, 3)

    return lin_2d, sin_2d, extent


def clean_points(
    points: np.ndarray, sigma: int = 2, fnc: Callable = spstats.norm
) -> np.ndarray:
    """Creates a numpy array for slicing that removes all outliers from the
    given points array. The outliers are defined by the standard deviation of
    the given statistical function which can be any scipy statistics function
    that supports standard deviations.

    Arguments:
        points -- The array of points to clean

    Keyword Arguments:
        sigma -- The amounts of standard deviations to use (default: {2})
        fnc -- The statistical function to use (default: {spstats.norm})

    Returns:
        A numpy array that can be used to slice other arrays
    """
    limit = sigma * fnc(*fnc.fit(points)).std()
    slc = np.nonzero(points > -limit) and np.nonzero(points < limit)
    return slc


def get_linfit_each_timeseries(data: dict) -> np.ndarray:
    """Fits each timeseries in the data dictionary. Suitable only for small
    datasets. Good for creating scatter plots.

    Arguments:
        data -- data -- Data dictionary according to u4py standard (e.g. read from h5)

    Returns:
        An array containing all linear fits in the dictionary.
    """

    lintrend = []
    time = data["time"]
    for y in data["timeseries"]:
        slc = np.nonzero(np.isfinite(y))
        time_slice = time[slc]
        y = y[slc]
        time_days = np.linspace(0, len(time_slice) * 6, len(time_slice))

        lin_popt, lin_pcov = spopt.curve_fit(
            u4other.poly1,
            time_days,
            y,
        )
        lintrend.append(lin_popt[0] * 365)
    return lintrend


def full_inv_single_comp(x: np.ndarray, *args) -> np.ndarray:
    """Returns a full fit including (semi-)annual sines and cosines:

    Structure of *args:
        [0]: yaxis-offset
        [1]: linear trend
        [2]: semi-annual sine
        [3]: semi-annual cosine
        [4]: annual sine
        [5]: annual cosine
    Arguments:
        x -- The x axis
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

    Arguments:
        x -- The x axis (datetime).
        inv_results -- A tuple of the inversion results.

    Keyword Arguments:
        dir -- The direction of the inversion results. Options are: "EW",
        "NS", "UD" (default {"UD"})

    Returns:
        The forward model for the specified direction
    """
    directions = {  # Index where to start looking for the right components
        "EW": 0,
        "NS": 6,
        "UD": 12,
    }
    ii = directions[dir]
    args = inv_results[ii : ii + 6]
    time = np.array([u4convert.get_floatyear(t) for t in x])
    y = full_inv_single_comp(time, *args)
    return y
