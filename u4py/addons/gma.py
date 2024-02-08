"""
Ground Motion Analyzer Addon

Utilizes a variance based filter to find ground motion anomalies in the
dataset.
"""

import logging
import os
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as skmeasure
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

import u4py.analysis.spatial as u4spatial
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff


def get_gma_results(
    psi_fpath: os.PathLike,
    processing_path: os.PathLike,
    cellsize: int,
    min_mean: float,
    max_var: float,
    crs: str,
    overwrite: bool = False,
) -> os.PathLike:
    """Looks for GMA-results in the path, if not creates them using the parameters given.

    :param psi_fpath: The path to the PSI-files.
    :type psi_fpath: os.PathLike
    :param processing_path: The folder where to store the tiffs.
    :type processing_path: os.PathLike
    :param cellsize: The cellsize of the GMA
    :type cellsize: int
    :param min_mean: The minimum velocity for the GMA
    :type min_mean: float
    :param max_var: The maximum variance for the GMA
    :type max_var: float
    :param crs: The coordinate system of the input data.
    :type crs: str
    :param overwrite: Whether to overwrite the existing tiff, defaults to False
    :type overwrite: bool, optional
    :return: The path to the GMA-results as tiff.
    :rtype: os.PathLike
    """
    gma_path = os.path.join(
        processing_path,
        f"GMA_results_cell={cellsize}_minmean={min_mean}_maxvar={max_var}.tif",
    )
    if (not os.path.exists(gma_path)) or overwrite:
        logging.info("No GMA-results found, calculating...")
        data = u4sql.table_to_dict(psi_fpath, "vertikal", get_timeseries=False)
        gma_grid = u4spatial.hotspots_GroundMotionAnalyzer(
            data, cellsize=cellsize, min_mean=min_mean, max_var=max_var
        )
        extent = (
            np.min(data["x"]) + cellsize,
            np.max(data["x"]) + cellsize,
            np.min(data["y"]) + cellsize,
            np.max(data["y"]) + cellsize,
        )
        gma_grid = np.rot90(gma_grid, 3)
        gma_grid = np.fliplr(gma_grid)

        logging.info("Saving GMA-results to file")
        u4tiff.ndarray_to_geotiff(
            gma_grid, extent, gma_path, crs=crs, compress="lzw"
        )
    else:
        logging.info("GMA-results found.")
    return gma_path


def hotspots_GroundMotionAnalyzer(
    data: dict, cellsize: int, min_mean: float, max_var: float
) -> Tuple[Figure, Axes, PolyCollection]:
    logging.info("Running GroundMotionAnalyzer")

    # Slice data with minimum velocity and maximum variance
    slc = (np.abs(data["mean_vel"]) >= min_mean) & (
        data["var_mean_vel"] < max_var
    )
    for k in data.keys():
        data[k] = data[k][slc]

    # Create a histogram counting the values in a regular grid.
    x_edges = np.arange(
        np.min(data["x"]), np.max(data["x"]) + cellsize, cellsize
    )
    y_edges = np.arange(
        np.min(data["y"]), np.max(data["y"]) + cellsize, cellsize
    )
    fig, ax = plt.subplots()
    hist_grid, _, _, _ = ax.hist2d(
        data["x"], data["y"], bins=(x_edges, y_edges), cmin=1, alpha=0.75
    )
    plt.close(fig)

    # Create gridded numpy array and do a blockwise reduction using sums
    grid = scattered_to_gridded(data["x"], data["y"], np.abs(data["mean_vel"]))
    red_grid = skmeasure.block_reduce(
        grid, block_size=int(cellsize / 50), func=np.nansum
    )
    red_grid[red_grid == 0] = np.nan
    gma_grid = np.log10(hist_grid * red_grid)

    return gma_grid


def scattered_to_gridded(
    x: Iterable, y: Iterable, values: Iterable, dx: float = 50, dy: float = 50
) -> np.ndarray:
    """Converts x, y, z values to a gridded numpy array.

    :param x: 1D array of x coordinates
    :type x: Iterable
    :param y: 1D array of y coordinates
    :type y: Iterable
    :param values: 1D array of corresponding z values
    :type values: Iterable
    :return: 2D numpy array representing the gridded data
    :rtype: np.ndarray
    """
    logging.info("Converting scattered data to grid.")
    xr = np.arange(np.min(x), np.max(x) + dx, dx)
    yr = np.arange(np.min(y), np.max(y) + dy, dy)
    xi = np.searchsorted(xr, x, side="left")
    yi = np.searchsorted(yr, y, side="left")
    grid = np.ones((len(xr), len(yr))) * np.nan
    grid[xi, yi] = values

    return grid


def hotspots_hexbin(
    vals: np.ndarray,
    coords: dict,
    thresh: float | Iterable,
    nbins: int = 3,
    min_count: int = 3,
) -> Tuple[Figure, Axes, PolyCollection]:
    """Uses matplotlibs hexbin to create a list of hotspots.

    These are areas in the data that are above the threshold and cover at least `nbins` x `(dx, dy)`.

    :param vals: A numpy array with the values to bin.
    :type vals: np.ndarray
    :param coords: A dictionary with minimum x, y and dx, dy.
    :type coords: dict
    :param thresh: Either a single threshold or an iterable with two thresholds. If it is iterable values below the lower threshold and above the higher threshold are detected as hotspots.
    :type thresh: float | Iterable
    :param nbins: The minimum area in n x (dx, dy) that a hotspot has to cover for detection, defaults to 3
    :type nbins: int
    :param min_count: The minimum number of psi in a bin to be detected, defaults to 3
    :return: The figure, axes and hexbin collection for further analysis.
    :rtype: Tuple[Figure, Axes, PolyCollection]
    """
    logging.info("Running HexagonalBinning")
    # Create grid for binning
    r, c = vals.shape
    x = np.arange(coords["x"], coords["x"] + (c * coords["dx"]), coords["dx"])
    y = np.arange(coords["y"], coords["y"] + (r * coords["dy"]), coords["dy"])
    xx, yy = np.meshgrid(x, y)

    # Thresholding
    if isinstance(thresh, Iterable):
        vals[(vals < np.max(thresh)) & (vals > np.min(thresh))] = np.nan
    else:
        vals[vals < thresh] = np.nan
    xx_real = xx[np.isfinite(vals)]
    yy_real = yy[np.isfinite(vals)]

    # Binning
    fig, ax = plt.subplots(figsize=(6, 10), dpi=150)
    h = ax.hexbin(
        xx_real,
        yy_real,
        gridsize=(int(c / nbins), int(r / nbins)),
        mincnt=min_count,
        alpha=0.75,
    )

    return (fig, ax, h)
