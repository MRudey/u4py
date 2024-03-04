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
from tqdm import tqdm

import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff


def get_gma_results(
    psi_fpath: os.PathLike,
    processing_path: os.PathLike,
    cellsize: int | list[int],
    min_mean: float | list[float],
    max_var: float | list[float],
    crs: str,
    overwrite: bool = False,
) -> os.PathLike:
    """Gets GMA results for given psi file using the parameters given. Any of
    the parameters `cellsize`, `min_mean` or `max_var` can be an iterable of
    values. Then the GMA results are computed for each combination of them.

    :param psi_fpath: The path to the PSI-files.
    :type psi_fpath: os.PathLike
    :param processing_path: The folder where to store the tiffs.
    :type processing_path: os.PathLike
    :param cellsize: The cellsize of the GMA
    :type cellsize: int | list[int]
    :param min_mean: The minimum velocity for the GMA
    :type min_mean: float | list[float]
    :param max_var: The maximum variance for the GMA
    :type max_var: float | list[float]
    :param crs: The coordinate system of the input data.
    :type crs: str
    :param overwrite: Whether to overwrite the existing tiff, defaults to False
    :type overwrite: bool, optional
    :return: The path to the GMA-results as tiff.
    :rtype: os.PathLike
    """
    gma_folder = os.path.join(processing_path, "GMA_results")
    os.makedirs(gma_folder, exist_ok=True)
    if not isinstance(cellsize, Iterable):
        cellsize = [cellsize]
    if not isinstance(min_mean, Iterable):
        min_mean = [min_mean]
    if not isinstance(max_var, Iterable):
        max_var = [max_var]

    params = []
    for cs in cellsize:
        for mm in min_mean:
            for mv in max_var:
                params.append((cs, mm, mv))
    data = u4sql.table_to_dict(
        psi_fpath, "vertikal", get_timeseries=False, recalculate_stats=True
    )
    for par in tqdm(params, desc="Calculating GMA", leave=False):
        load_gma_results(data, gma_folder, overwrite, crs, *par)


def load_gma_results(
    data: dict,
    gma_folder: os.PathLike,
    overwrite: bool,
    crs: bool,
    cellsize: int,
    min_mean: float,
    max_var: float,
) -> os.PathLike:
    """Looks for GMA-results in the path, if not creates them using the parameters given.

    :param data: The data loaded from the psi file.
    :type data: dict
    :param gma_folder: The folder where to store the results
    :type gma_folder: os.PathLike
    :param overwrite: Whether to overwrite the existing files
    :type overwrite: bool
    :param crs: The coordinate system of the data.
    :type crs: bool
    :param cellsize: The cellsize of the GMA
    :type cellsize: int
    :param min_mean: The minimum velocity for the GMA
    :type min_mean: float
    :param max_var: The maximum variance for the GMA
    :type max_var: float
    :return: The path to the generated tiff file
    :rtype: os.PathLike
    """
    gma_path = os.path.join(
        gma_folder,
        f"GMA_results_cell={cellsize}_minmean={min_mean}_maxvar={max_var}.tif",
    )
    if (not os.path.exists(gma_path)) or overwrite:
        logging.info("No GMA-results found, calculating...")
        gma_grid, extent = hotspots_GroundMotionAnalyzer(
            data, cellsize=cellsize, min_mean=min_mean, max_var=max_var
        )
        if len(gma_grid) > 0:
            logging.info("Saving GMA-results to file")
            u4tiff.ndarray_to_geotiff(
                gma_grid, extent, gma_path, crs=crs, compress="lzw"
            )
        else:
            logging.info(
                f"No GMA results for cell={cellsize}, minmean={min_mean}, maxvar={max_var}"
            )
            return []
    else:
        logging.info("GMA-results found.")
    return gma_path


def hotspots_GroundMotionAnalyzer(
    data: dict, cellsize: int, min_mean: float, max_var: float
) -> Tuple[np.ndarray, tuple]:
    """Computes deformation hotspots using the GroundMotionAnalyzer

    :param data: The data loaded from the psi file.
    :type data: dict
    :param cellsize: The cellsize of the GMA
    :type cellsize: int
    :param min_mean: The minimum velocity for the GMA
    :type min_mean: float
    :param max_var: The maximum variance for the GMA
    :type max_var: float
    :return: The computed grid and its extend as tuple.
    :rtype: Tuple[np.ndarray, tuple]

    Returns an empty list if no hotspots were found.
    """
    logging.info("Running GroundMotionAnalyzer")

    # Slice data with minimum velocity and maximum variance
    slc = (np.abs(data["mean_vel"]) >= min_mean) & (
        data["var_mean_vel"] ** 2 < max_var
    )
    slc_data = dict()
    for k in data.keys():
        try:
            slc_data[k] = data[k][slc]
        except TypeError:
            pass
    logging.debug(f"{slc_data['x'].shape}")
    if len(slc_data["x"]) > 0:
        # Create gridded numpy array and do a blockwise reduction using sums
        grid = scattered_to_gridded(
            slc_data["x"], slc_data["y"], np.abs(slc_data["mean_vel"])
        )
        red_grid = skmeasure.block_reduce(
            grid, block_size=int(cellsize / 50), func=np.nansum
        )
        r, c = red_grid.shape

        # Create a histogram counting the values in a regular grid.
        x_edges = np.linspace(
            np.min(slc_data["x"]), np.max(slc_data["x"]), r + 1
        )
        y_edges = np.linspace(
            np.min(slc_data["y"]), np.max(slc_data["y"]), c + 1
        )
        fig, ax = plt.subplots()
        hist_grid, _, _, _ = ax.hist2d(
            slc_data["x"],
            slc_data["y"],
            bins=(x_edges, y_edges),
            cmin=1,
            alpha=0.75,
        )
        plt.close(fig)

        red_grid[red_grid == 0] = np.nan
        gma_grid = np.log10(hist_grid * red_grid)
        extent = (
            np.min(slc_data["x"]),
            np.max(slc_data["x"]) + cellsize,
            np.min(slc_data["y"]),
            np.max(slc_data["y"]) + cellsize,
        )
        gma_grid = np.rot90(gma_grid, 3)
        gma_grid = np.fliplr(gma_grid)
        return gma_grid, extent
    else:
        return []


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
