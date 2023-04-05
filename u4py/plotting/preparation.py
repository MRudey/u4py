""" Contains functions to modify or reformat data for plotting """

from typing import Callable

import numpy as np
import scipy.ndimage as spimg
import scipy.optimize as spopt
import scipy.stats as spstats

import u4py.analysis.other as u4other


def convert_results_for_grid(results, chunk_size):
    """Returns the loaded data prepared for gridding"""
    if not chunk_size:
        chunk_size = 250
    xmids = []
    ymids = []
    slope = []
    season = []
    for xmid, ymid, components in results:
        xmids.append(xmid)
        ymids.append(ymid)
        if components is not None:
            if len(components) < 5:
                slope.append(components[0] * 365.25)
                season.append(np.abs(components[1]))
            else:
                slope.append(components[13])
                season.append(np.abs(components[14]))
        else:
            season.append(np.nan)
    return xmids, ymids, slope, season, chunk_size


def make_gridded_data(xmids, ymids, slope, season, chunk_size):
    minx = np.min(xmids)
    maxx = np.max(xmids) + chunk_size
    miny = np.min(ymids)
    maxy = np.max(ymids) + chunk_size
    extent = (minx, maxx, miny, maxy)
    x = np.arange(minx, maxx, chunk_size)
    y = np.arange(miny, maxy, chunk_size)

    XX, YY = np.meshgrid(x, y)
    slp_2d = np.ones_like(XX) * np.nan
    sea_2d = np.ones_like(XX) * np.nan

    for xi, yi, sl, se in zip(xmids, ymids, slope, season):
        xn = int((xi - minx) / chunk_size)
        yn = int((yi - miny) / chunk_size)
        slp_2d[yn, xn] = sl
        sea_2d[yn, xn] = se

    slp_2d = spimg.median_filter(slp_2d, 3)
    sea_2d = spimg.median_filter(sea_2d, 3)

    return slp_2d, sea_2d, extent


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

    slope = []
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
        slope.append(lin_popt[0] * 365)
    return slope
