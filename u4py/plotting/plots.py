"""
Contains functions with ready made plots. This module uses axis functions defined in :func:`u4py.plotting.axes` to create more complicated plots. The functions also do some processing and other data modification. Each function should contain a `save_path` if possible where the output figure is saved. If none is given the figure is shown interactively, otherwise it is saved and then destroyed.
"""

from __future__ import annotations

import os
from typing import Iterable

import contextily
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np

import u4py.addons.gma as u4gma
import u4py.io.tiff as u4tiff
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt


def plot_inversion_results(
    time: np.ndarray,
    time2: np.ndarray = np.array([]),
    data: dict = dict(),
    inversion_results: dict = dict(),
    save_path: os.PathLike = None,
    single_dim: bool = False,
    unit: str = "mm",
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
    :param single_dim: If the input data is single dimensional reduce to one axis, defaults to False
    :type single_dim: bool, optional
    :param unit: The unit of the input signal, defaults to "mm"
    :type unit: str, optional
    """
    if single_dim:
        fig, axes = plt.subplots(figsize=(10, 5))
        axes.plot(time[0], data["dataE"], ".", label="Data")
        axes.plot(
            time[0],
            data["ori_dhat_data"]["dhatE"],
            color="C1",
            label="Inversion",
        )
    else:
        fig, axes = plt.subplots(ncols=2, figsize=(10, 5), sharex=True)

    if not single_dim:
        ew_mov = inversion_results["matrix_ori"][1]
        ud_mov = inversion_results["matrix_ori"][13]
        ux = np.unique(time[0])
        y = [
            np.nanmedian(data["dataE"][np.argwhere(time[0] == uu)])
            for uu in ux
        ]
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
            ux,
            yed,
            yep,
            color="C0",
            alpha=0.5,
            edgecolor=None,
            label="95% range",
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
        y = [
            np.nanmedian(data["dataU"][np.argwhere(time[0] == uu)])
            for uu in ux
        ]
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
        axes[1].fill_between(
            ux, yed, yep, color="C0", alpha=0.5, edgecolor=None
        )
        axes[1].fill_between(
            ux, yed2, yep2, color="C0", alpha=0.5, edgecolor=None
        )
        axes[1].plot(time[0], data["ori_dhat_data"]["dhatU"])
        if time2.size > 0:
            axes[1].plot(time2[0], data["dhat_data"]["dhatU"])
        axes[1].annotate(
            f"Hebung/Senkung = {ud_mov:.2} {unit}/yr",
            (0.05, 0.05),
            xycoords="axes fraction",
        )
    # axes[1][1].plot(time[0], data["ori_dhat_data"]["dhatU"], "C1")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        return fig, axes


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


def plot_hotspots(
    tif_file_path: os.PathLike,
    thresh: float | Iterable,
    nbins: int = 3,
    min_count: int = 3,
    output_filepath: os.PathLike = "",
    title: str = "",
) -> list:
    """Loads the data from a tiff file and detects hotspots.

    :param tif_file_path: The path to the Tiff file.
    :type tif_file_path: os.PathLike
    :param thresh: Either a single threshold or an iterable with two thresholds. If it is iterable values below the lower threshold and above the higher threshold are detected as hotspots.
    :type thresh: float | Iterable
    :param nbins: The minimum area in n x (dx, dy) that a hotspot has to cover for detection, defaults to 3
    :type nbins: int
    :param min_count: The minimum number of psi in a bin to be detected, defaults to 3
    :type min_count: int
    :param output_filepath: The output path of the figure, defaults to ""
    :type output_filepath: os.PathLike, optional
    :param title: The title to put on the figure, defaults to ""
    :type title: str, optional
    :return: A list of shapely polygons for plotting and exporting to files.
    :rtype: list
    """

    # Load data
    coords, vals, crs = u4tiff.extract_xyz_tiff(tif_file_path)

    fig, ax, h = u4gma.hotspots_hexbin(vals, coords, thresh, nbins, min_count)

    if output_filepath:
        fpath_woex = os.path.splitext(output_filepath)[0]
        r, c = vals.shape
        ax.set_xlim(coords["x"], coords["x"] + (c * coords["dx"]))
        ax.set_ylim(coords["y"], coords["y"] + (r * coords["dy"]))
        u4ax.add_basemap(ax=ax, crs=crs)
        u4plotfmt.map_style(ax, divisor=25000, crs=crs)
        res = coords["dx"] * nbins
        if isinstance(thresh, Iterable):
            thresh_str = f"<{np.min(thresh)} or >{np.max(thresh)}"
        else:
            thresh_str = f"{thresh}"
        ax.annotate(
            f"Hexgrid: {res} m\nThreshold: {thresh_str} mm\nMin. #PSI: {min_count}",
            (0, -0.1),
            xycoords="axes fraction",
            fontsize="small",
            annotation_clip=False,
        )
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(f"{fpath_woex}.pdf")
        fig.savefig(f"{fpath_woex}.png")

    return (h.get_offsets(), h.get_array(), crs)


def plot_GroundMotionAnalyzer(
    psi_fpath: os.PathLike,
    processing_path: os.PathLike,
    cellsize: int = 500,
    min_mean: float = 2,
    max_var: float = 1,
    output_filepath: os.PathLike = "",
    title: str = "",
    crs: str = "EPSG:32632",
    overwrite: bool = False,
):
    """Creates and plots the results for the GroundMotionAnalyzer

    :param psi_fpath: The path to the gpkg file containing the PSI data.
    :type psi_fpath: os.PathLike
    :param processing_path: The folderpath where to save the results as a tiff file.
    :type processing_path: os.PathLike
    :param cellsize: The cellsize of the ground motion analyzer in meters, defaults to 500
    :type cellsize: int, optional
    :param min_mean: The minimum mean velocity for the detection, defaults to 2
    :type min_mean: float, optional
    :param max_var: The maximum variance for the detection, defaults to 1
    :type max_var: float, optional
    :param output_filepath: The output file path of the figure, if none is given no plot is produced, defaults to ""
    :type output_filepath: os.PathLike, optional
    :param title: The title for the plot, defaults to ""
    :type title: str, optional
    :param crs: The coordinate system of the data, defaults to "EPSG:32632"
    :type crs: str, optional
    :param overwrite: Whether to overwrite the existing results, defaults to False
    :type overwrite: bool, optional
    """
    gma_tile_path = u4gma.get_gma_results(
        psi_fpath,
        processing_path,
        cellsize,
        min_mean,
        max_var,
        crs,
        overwrite=overwrite,
    )
    if output_filepath:
        fig, ax = plt.subplots(figsize=(6, 10), dpi=150)
        u4ax.add_tile(gma_tile_path, ax=ax, cmap="Reds", vm=(-2, -1), zorder=3)
        fpath_woex = os.path.splitext(output_filepath)[0]
        u4ax.add_basemap(ax=ax, crs=crs)
        u4plotfmt.map_style(ax, divisor=25000, crs=crs)
        ax.annotate(
            f"Cellsize: {cellsize} m\nThreshold: >={min_mean} mm/a\nMax. Variance: {max_var}",
            (0, -0.1),
            xycoords="axes fraction",
            fontsize="small",
            annotation_clip=False,
        )
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(f"{fpath_woex}.pdf")
        fig.savefig(f"{fpath_woex}.png")
