"""
Contains functions with ready made plots. This module uses axis functions defined in :func:`u4py.plotting.axes` to create more complicated plots. The functions also do some processing and other data modification. Each function should contain a `save_path` if possible where the output figure is saved. If none is given the figure is shown interactively, otherwise it is saved and then destroyed.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Iterable, Tuple

import contextily
import geopandas as gp
import matplotlib.artist as martist
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import u4py.addons.gma as u4gma
import u4py.addons.web_services as u4web
import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.psi as u4psi
import u4py.io.tiff as u4tiff
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt

warnings.filterwarnings("ignore", module="matplotlib")

IMAGE_DPI = 150  # DPI setting for plots
IMAGE_FORMATS = ["pdf"]  # List of filetypes to export for figures
IS_HLNUG = False


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
    logging.info("Plotting inversion results")
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
    logging.info("Plotting gridded data")
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
    logging.info("Detecting and Plotting Hotspots")
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
        for ftype in IMAGE_FORMATS:
            fig.savefig(f"{fpath_woex}.{ftype}")

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
    logging.info("Running and plotting GroundMotionAnalyzer")
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
        fig, ax = plt.subplots(figsize=(6, 10), dpi=IMAGE_DPI)
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

        for ftype in IMAGE_FORMATS:
            fig.savefig(f"{fpath_woex}.{ftype}")
        plt.close(fig)


def plot_site_statistics(
    res: dict, x_key: str, y_key: str, folder_path: os.PathLike
):
    """Plots statistics of a given site that was classified using `u4py.analysis.classify`.

    :param res: The result dictionary.
    :type res: dict
    :param x_key: The key for the x-axis (must be in `res.keys()`)
    :type x_key: str
    :param y_key: The key for the y-axis (must be in `res.keys()`)
    :type y_key: str
    :param folder_path: The output folder.
    :type folder_path: os.PathLike
    """
    logging.info("Calculating and plotting site statistics.")
    if x_key in res.keys() and y_key in res.keys():
        site_dir = os.path.join(folder_path, f"Site_{res['group']}")
        os.makedirs(site_dir, exist_ok=True)
        # Plot landuse statistics
        fig, ax = default_figure()
        ax.barh(
            res[x_key],
            res[y_key],
        )
        fig.tight_layout()
        for ftype in IMAGE_FORMATS:
            fig.savefig(
                os.path.join(
                    site_dir, f"{x_key}_vs_{y_key}_{res['group']}.{ftype}"
                )
            )
        plt.close(fig)
    else:
        raise KeyError(
            f"At least one of the keys: {x_key} or {y_key} not found in the results dictionary."
        )


def plot_shape(
    sub_set: gp.GeoDataFrame,
    sub_set_hull: gp.GeoDataFrame,
    roads: gp.GeoDataFrame,
    slope_str: str,
    group: int,
    save_folder: os.PathLike,
):
    """Plots the given subset and some additional data.

    :param sub_set: The subset of shapes for a specific group.
    :type sub_set: gp.GeoDataFrame
    :param sub_set_hull: The area surrounding the subset.
    :type sub_set_hull: gp.GeoDataFrame
    :param roads: The roads loaded from openstreetmap data.
    :type roads: gp.GeoDataFrame
    :param slope_str: The average slope in each of the polygons of the subset.
    :type slope_str: str
    :param group: The group number.
    :type group: int
    :param save_folder: The folder where to store the output figure.
    :type save_folder: os.PathLike
    """
    logging.info(f"Plotting a subset of shapes of group {group:05}.")
    fig, ax = default_figure()
    # shp_gdf.plot(ax=ax, column="groups", cmap="tab20")
    sub_set.plot(
        ax=ax,
        column="slope_mean",
        ec="w",
        vmin=0,
        vmax=90,
        legend=True,
        legend_kwds={"label": "Slope (deg)"},
    )
    sub_set_hull.plot(ax=ax, fc="None", ec="k")
    roads.plot(ax=ax, column="fclass")
    ax.axis("equal")
    u4plotfmt.map_style(ax)
    ax.annotate(slope_str, (0.025, 0.025), xycoords="axes fraction", zorder=4)
    fig.tight_layout()
    u4ax.add_basemap(
        ax=ax,
        crs=sub_set.crs,  # source=contextily.providers.CartoDB.Positron
    )
    for ftype in IMAGE_FORMATS:
        fig.savefig(os.path.join(save_folder, f"Site_{group}.{ftype}"))
    plt.close(fig)


def default_figure():
    return plt.subplots(figsize=(7, 7), dpi=100)


def geology_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    contour_path: os.PathLike,
    legend_path: os.PathLike,
    shp_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
    use_internal: bool = True,
):
    """Creates a map of the geological features around the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param legend_path: The path where the legend is found.
    :type legend_path: os.PathLike
    :param shp_path: The path where query data for HLNUG/GeoServer Data is found.
    :type shp_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """

    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_GK25.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting geological map of group {row[1].group:05}.")

    # Make geodataframe and add to plot
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0", zorder=5, linewidth=3)
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Plot data
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    if use_internal:
        try:
            bounds = region.bounds.iloc[0]
            geology_data = u4web.query_internal(
                "gk25-hessen:GK25_f_GK3",
                region=[bounds.minx, bounds.miny, bounds.maxx, bounds.maxy],
                region_crs=region.crs,
                suffix=f"{row[1].group:05}",
                out_folder=shp_path,
            )
        except TimeoutError:
            logging.debug("Internal Server timed out, using fallback.")
            geology_data = u4web.query_hlnug(
                "geologie/gk25/MapServer",
                "Geologie (Kartiereinheiten)",
                region=region,
                suffix=f"_{row[1].group:05}",
                out_folder=shp_path,
            )
    else:
        geology_data = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Geologie (Kartiereinheiten)",
            region=region,
            suffix=f"_{row[1].group:05}",
            out_folder=shp_path,
        )
    if not geology_data.empty:
        # Get symbology and fill legend entry lists
        if "TKEH" in geology_data.keys():  # For data from HLNUG
            leg_list = geology_data["TKEH"].to_list()
        else:  # For data from internal geoserver
            leg_list = [
                int(f"{tknr}{eh:03}")
                for tknr, eh in zip(
                    geology_data["tknr"].to_list(),
                    geology_data["einheit"].to_list(),
                )
            ]

        leg_handles = []
        leg_labels = []

        leg_handles, leg_labels = u4ax.add_patch_legend_entry(
            leg_handles,
            leg_labels,
            "Bereich der Anomalie",
            facecolor="None",
            edgecolor="C0",
            linewidth=3,
        )

        leg_dict, lgh_from_file, lglb_from_file = (
            u4plotfmt.get_style_from_legend(leg_list, legend_path)
        )
        leg_handles.extend(lgh_from_file)
        leg_labels.extend(lglb_from_file)

        # Plot geological map
        u4ax.add_hlnug_shapes(
            ax=ax,
            geometries=geology_data.geometry.to_list(),
            leg_dict=leg_dict,
        )

        # Plot other anomalies
        if not IS_HLNUG:
            # For non HLNUG data use the thresholded contours
            u4ax.add_gpkg_data_in_axis(
                contour_path,
                ax=ax,
                table="thresholded_contours_all_shapes",
                edgecolor="k",
                facecolor="None",
                linewidth=1,
            )
        else:
            # For HLNUG data use the
            u4ax.add_gpkg_data_in_axis(
                contour_path,
                ax=ax,
                table="Classified_Shapes",
                edgecolor="k",
                facecolor="None",
                linewidth=1,
            )

        # Load tectonic data
        if use_internal:
            try:
                fault_data = u4web.query_internal(
                    "gk25-hessen:GK25_l-tek_GK3",
                    region=[
                        bounds.minx,
                        bounds.miny,
                        bounds.maxx,
                        bounds.maxy,
                    ],
                    region_crs=region.crs,
                    suffix=f"{row[1].group:05}",
                    out_folder=shp_path,
                )
            except TimeoutError:
                logging.debug("Internal Server timed out, using fallback.")
                fault_data = u4web.query_hlnug(
                    "geologie/gk25/MapServer",
                    "Tektonik (Liniendaten)",
                    region=region,
                    suffix=f"_{row[1].group:05}",
                    out_folder=shp_path,
                )
        else:
            fault_data = u4web.query_hlnug(
                "geologie/gk25/MapServer",
                "Tektonik (Liniendaten)",
                region=region,
                suffix=f"_{row[1].group:05}",
                out_folder=shp_path,
            )
        if not fault_data.empty:
            leg_handles, leg_labels = u4ax.add_fault_data(
                fault_data=fault_data,
                leg_handles=leg_handles,
                leg_labels=leg_labels,
                ax=ax,
            )

        hydro_pts = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Hydrogeologie (Punktdaten)",
            region=region,
            suffix=f"_hygpts_{row[1].group:05}",
            out_folder=shp_path,
        )
        if not hydro_pts.empty:
            leg_handles, leg_labels = u4ax.add_hydrological_points(
                hydro_pts=hydro_pts,
                leg_handles=leg_handles,
                leg_labels=leg_labels,
                ax=ax,
            )

        soggy_areas = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Nassstellen",
            region=region,
            suffix=f"_nass_{row[1].group:05}",
            out_folder=shp_path,
        )
        if not soggy_areas.empty:
            leg_handles, leg_labels = u4ax.add_soggy_areas(
                soggy_areas=soggy_areas,
                leg_handles=leg_handles,
                leg_labels=leg_labels,
                ax=ax,
            )

        water_surface = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Gewässer",
            region=region,
            suffix=f"_water_{row[1].group:05}",
            out_folder=shp_path,
        )
        if not water_surface.empty:
            leg_handles, leg_labels = u4ax.add_water_surface(
                water_surface=water_surface,
                leg_handles=leg_handles,
                leg_labels=leg_labels,
                ax=ax,
            )

        geo_pts = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Geologie (Punktdaten)",
            region=region,
            suffix=f"_gepts_{row[1].group:05}",
            out_folder=shp_path,
        )
        if not geo_pts.empty:
            leg_handles, leg_labels = u4ax.add_geo_pts(
                geo_pts=geo_pts,
                leg_handles=leg_handles,
                leg_labels=leg_labels,
                ax=ax,
            )

        drill_sites = u4web.query_hlnug(
            "geologie/bohrdatenportal/MapServer",
            "Archivbohrungen, Endteufe [m]",
            region=region,
            suffix=f"_bohr_{row[1].group:05}",
            out_folder=shp_path,
        )
        if not drill_sites.empty:
            leg_handles, leg_labels = u4ax.add_drill_sites(
                drill_sites=drill_sites,
                leg_handles=leg_handles,
                leg_labels=leg_labels,
                ax=ax,
            )

        # Formatting and other stuff
        u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
        u4ax.add_basemap(
            ax=ax,
            crs=geology_data.crs,
            source=contextily.providers.TopPlusOpen.Grey,
        )

        for ftype in IMAGE_FORMATS:
            fig.savefig(
                os.path.join(output_path, f"{row[1].group:05}_GK25.{ftype}")
            )

        # Create legend
        plot_legend(leg_handles, leg_labels, output_path, row[1].group, "GK25")

    plt.close(fig)


def hydrogeology_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    contour_path: os.PathLike,
    legend_path: os.PathLike,
    shp_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates a map of hydrogeological units in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param legend_path: The path where the legend is found.
    :type legend_path: os.PathLike
    :param shp_path: The path where query data for HLNUG/GeoServer Data is found.
    :type shp_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """
    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_HUEK200.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(
            f"Plotting hydrogeological map of group {row[1].group:05}."
        )

    # Make geodataframe and add to plot
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0", zorder=5, linewidth=3)
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Plot data
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    hydro_units_data = u4web.query_hlnug(
        "geologie/huek200/MapServer",
        "Hydrogeologische Einheiten",
        region=region,
        suffix=f"{row[1].group:05}",
        out_folder=shp_path,
    )

    if len(hydro_units_data) > 0:
        # Get symbology and fill legend entry lists
        leg_dict, lgh_from_file, lglb_from_file = (
            u4plotfmt.get_style_from_legend(
                hydro_units_data["L_HE_B_KUE"].to_list(), legend_path
            )
        )
        leg_handles = []
        leg_labels = []
        leg_handles, leg_labels = u4ax.add_patch_legend_entry(
            leg_handles,
            leg_labels,
            "Bereich der Anomalie",
            facecolor="None",
            edgecolor="C0",
            linewidth=3,
        )
        leg_handles.extend(lgh_from_file)
        leg_labels.extend(lglb_from_file)

        # Plot hydrogeological map
        u4ax.add_hlnug_shapes(
            hydro_units_data.geometry.to_list(), leg_dict=leg_dict, ax=ax
        )

        # Plot other anomalies
        if not IS_HLNUG:
            # For non HLNUG data use the thresholded contours
            u4ax.add_gpkg_data_in_axis(
                contour_path,
                ax=ax,
                table="thresholded_contours_all_shapes",
                edgecolor="k",
                facecolor="None",
                linewidth=1,
            )
        else:
            # For HLNUG data use the
            u4ax.add_gpkg_data_in_axis(
                contour_path,
                ax=ax,
                table="Classified_Shapes",
                edgecolor="k",
                facecolor="None",
                linewidth=1,
            )

        # Formatting and other stuff
        u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
        u4ax.add_basemap(
            ax=ax,
            crs=hydro_units_data.crs,
            source=contextily.providers.TopPlusOpen.Grey,
        )
        for ftype in IMAGE_FORMATS:
            fig.savefig(
                os.path.join(output_path, f"{row[1].group:05}_HUEK200.{ftype}")
            )

        # Create legend
        plot_legend(
            leg_handles, leg_labels, output_path, row[1].group, "HUEK200"
        )
    plt.close(fig)


def topsoil_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    contour_path: os.PathLike,
    legend_path: os.PathLike,
    shp_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates a map of topsoil units in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param legend_path: The path where the legend is found.
    :type legend_path: os.PathLike
    :param shp_path: The path where query data for HLNUG/GeoServer Data is found.
    :type shp_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """

    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_BFD50.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting topsoil map of group {row[1].group:05}.")

    # Make geodataframe and add to plot
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0", zorder=5, linewidth=3)
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Plot data
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    soil_data = u4web.query_hlnug(
        "boden/bfd50/MapServer",
        "BFD50_Bodenhauptgruppen",
        region=region,
        suffix=f"{row[1].group:05}",
        out_folder=shp_path,
    )

    if len(soil_data) > 0:
        if "UNTERGRUPPE" in soil_data.keys():  # SHP files have less chars...
            column = "UNTERGRUPPE"
        else:
            column = "UNTERGRUPP"
        leg_dict, lgh_from_file, lglb_from_file = (
            u4plotfmt.get_style_from_legend(
                soil_data[column].to_list(), legend_path
            )
        )
        leg_dict["alpha"] = [0.4 * a for a in leg_dict["alpha"]]
        leg_handles = []
        leg_labels = []
        leg_handles, leg_labels = u4ax.add_patch_legend_entry(
            leg_handles,
            leg_labels,
            "Bereich der Anomalie",
            facecolor="None",
            edgecolor="C0",
            linewidth=3,
        )
        leg_handles.extend(lgh_from_file)
        leg_labels.extend(lglb_from_file)

        # Plot soil map
        u4ax.add_hlnug_shapes(
            soil_data.geometry.to_list(), leg_dict=leg_dict, ax=ax
        )

        # Plot other anomalies
        if not IS_HLNUG:
            # For non HLNUG data use the thresholded contours
            u4ax.add_gpkg_data_in_axis(
                contour_path,
                ax=ax,
                table="thresholded_contours_all_shapes",
                edgecolor="k",
                facecolor="None",
                linewidth=1,
            )
        else:
            # For HLNUG data use the
            u4ax.add_gpkg_data_in_axis(
                contour_path,
                ax=ax,
                table="Classified_Shapes",
                edgecolor="k",
                facecolor="None",
                linewidth=1,
            )

        # Formatting and other stuff
        u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
        u4ax.add_basemap(
            ax=ax,
            crs=soil_data.crs,
            source=contextily.providers.TopPlusOpen.Grey,
        )
        for ftype in IMAGE_FORMATS:
            fig.savefig(
                os.path.join(output_path, f"{row[1].group:05}_BFD50.{ftype}")
            )

        # Create legend
        plot_legend(
            leg_handles, leg_labels, output_path, row[1].group, "BFD50"
        )
    plt.close(fig)


def satimg_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    contour_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates a satellite overview of the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """

    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_satimg.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting satellite image of group {row[1].group:05}.")

    # Make gdf and plot data
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Plot data
    u4ax.add_basemap(
        ax=ax, crs=crs, source=contextily.providers.Esri.WorldImagery
    )
    # Plot other anomalies
    if not IS_HLNUG:
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="thresholded_contours_all_shapes",
            edgecolor="w",
            facecolor="None",
            linewidth=2,
        )
    else:
        # For HLNUG data use the
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="Classified_Shapes",
            edgecolor="w",
            facecolor="None",
            linewidth=2,
        )

    # Formatting and save
    u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)

    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_satimg.{ftype}")
        )
    plt.close(fig)


def dem_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    dem_path: os.PathLike,
    contour_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
    shp_path: os.PathLike = "",
):
    """Creates a hillshade map of the digital elevation model in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param dem_path: The path where the dem data is found.
    :type dem_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """
    # Setting paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_dem.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(
            f"Plotting digital elevation model of group {row[1].group:05}."
        )

    # Loading data and plot it
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Add DEM in region
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    u4ax.add_dem(region, dem_path, ax=ax)

    # Plot other anomalies
    if not IS_HLNUG:
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="thresholded_contours_all_shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    else:
        # For HLNUG data use the
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="Classified_Shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
        if shp_path:
            drill_sites = u4web.query_hlnug(
                "geologie/bohrdatenportal/MapServer",
                "Archivbohrungen, Endteufe [m]",
                region=region,
                suffix=f"_bohr_{row[1].group:05}",
                out_folder=shp_path,
            )
            if not drill_sites.empty:
                u4ax.add_drill_sites(
                    drill_sites=drill_sites,
                    ax=ax,
                )
                leg_handles, leg_labels = u4ax.add_drill_sites(
                    drill_sites=drill_sites,
                    leg_handles=[],
                    leg_labels=[],
                    ax=ax,
                )
                ax.legend(leg_handles, leg_labels)

    # Formatting and other stuff
    u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_dem.{ftype}")
        )
    plt.close(fig)


def slope_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    dem_path: os.PathLike,
    contour_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates a slope map of the digital elevation model in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param dem_path: The path where the dem data is found.
    :type dem_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """

    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_slope.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting slope map of group {row[1].group:05}.")

    # Make geodataframe and add to plot
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Plot data
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    u4ax.add_slope(region, dem_path, ax=ax)
    # Plot other anomalies
    if not IS_HLNUG:
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="thresholded_contours_all_shapes",
            edgecolor="w",
            facecolor="None",
            linewidth=2,
        )
    else:
        # For HLNUG data use the
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="Classified_Shapes",
            edgecolor="w",
            facecolor="None",
            linewidth=2,
        )

    # Formatting and other stuff
    u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_slope.{ftype}")
        )
    plt.close(fig)


def aspect_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    dem_path: os.PathLike,
    contour_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates an aspect map of the digital elevation model in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param dem_path: The path where the dem data is found.
    :type dem_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """

    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_aspect.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting aspect map of group {row[1].group:05}.")

    # Make geodataframe and add to plot
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="k")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    u4ax.add_aspect(region, dem_path, ax=ax)
    # Plot other anomalies
    if not IS_HLNUG:
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="thresholded_contours_all_shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    else:
        # For HLNUG data use the
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="Classified_Shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    # Formatting and other stuff
    u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_aspect.{ftype}")
        )
    plt.close(fig)


def aspect_slope_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    dem_path: os.PathLike,
    contour_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates an aspect map of the digital elevation model in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param dem_path: The path where the dem data is found.
    :type dem_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """

    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(
                output_path, f"{row[1].group:05}_aspect_slope.{ftype}"
            )
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting aspect-slope map of group {row[1].group:05}.")

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="k")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    u4ax.add_aspect_slope(region, dem_path, ax=ax)
    # Plot other anomalies
    if not IS_HLNUG:
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="thresholded_contours_all_shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    else:
        # For HLNUG data use the
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="Classified_Shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )

    # Formatting and other stuff
    u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(
                output_path, f"{row[1].group:05}_aspect_slope.{ftype}"
            )
        )
    plt.close(fig)


def diffplan_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    diff_plan_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
):
    """Creates a map of the differences in surface elevation in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param diff_plan_path: The path where the differential data is found.
    :type diff_plan_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """
    # Setting Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_diffplan.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting difference map of group {row[1].group:05}.")

    # Loading Data
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)

    # Plotting
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Add diff plan in region
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    u4ax.add_diff_plan(region, diff_plan_path, ax=ax)

    # Format and save
    u4plotfmt.add_scalebar(ax=ax, width=2 * plot_buffer)
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_diffplan.{ftype}")
        )
    plt.close(fig)


def detailed_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    hlnug_path: os.PathLike,
    contour_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
    places_path: os.PathLike,
) -> Tuple[tuple, tuple]:
    """Makes a detailed overview map of the region including some geological features.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The folder where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for saving.
    :type suffix: str
    :param hlnug_path: The path where the HLNUG data is found.
    :type hlnug_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    :param places_path: The path where other shapefiles are found.
    :type places_path: os.PathLike
    """
    # Setup Paths
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)

    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_map.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(f"Plotting detailed map of group {row[1].group:05}.")

    # Make geodataframe and add to plot
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=IMAGE_DPI)
    shp_gdf.plot(ax=ax, fc="None", ec="C1", label="Region")
    shp_gdf.buffer(plot_buffer).plot(ax=ax, fc="None", ec="None")

    # Make plot full image size and fix axis to current extent
    u4plotfmt.full_screen_map(ax=ax)

    # Plot other anomalies
    if not IS_HLNUG:
        # Add additional geological data to plot.
        u4ax.add_gpkg_data_in_axis(
            hlnug_path,
            "rutschungen_mittelpunkte_2021_06_21",
            ax=ax,
            color="k",
            marker="$\Swarrow$",
            markersize=30,
            label="Rutschungen, Mittelpunkte",
        )
        u4ax.add_gpkg_data_in_axis(
            hlnug_path,
            "steinschlag_punkte",
            ax=ax,
            color="k",
            marker="$\therefore$",
            markersize=30,
            label="Steinschläge",
        )
        u4ax.add_gpkg_data_in_axis(
            hlnug_path,
            "Erdfaelle_merged",
            ax=ax,
            color="k",
            marker="$\odot$",
            markersize=30,
            label="Erdfälle",
        )
        u4ax.add_gpkg_data_in_axis(
            hlnug_path,
            "senkungsmulden",
            ax=ax,
            facecolor="None",
            edgecolor="k",
            linestyle=":",
            label="Senkungsmulden",
        )
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="thresholded_contours_all_shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    else:
        # For HLNUG data use the other shapes
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax,
            table="Classified_Shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
        handle_motorway = u4ax.add_gpkg_data_in_axis(
            places_path,
            ax=ax,
            table="gis_osm_roads_free_1",
            fclass="motorway",
        )
        handle_primary = u4ax.add_gpkg_data_in_axis(
            places_path,
            ax=ax,
            table="gis_osm_roads_free_1",
            fclass="primary",
        )
        handle_secondary = u4ax.add_gpkg_data_in_axis(
            places_path,
            ax=ax,
            table="gis_osm_roads_free_1",
            fclass="secondary",
        )
    ax.annotate(
        row[1].locations.replace(", ", "\n"),
        (0.99, 0.01),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
        zorder=10,
    )
    with warnings.catch_warnings():
        # Catches UserWarning for unsupported handles.
        warnings.simplefilter("ignore")
        h, _ = ax.get_legend_handles_labels()
    if not IS_HLNUG:
        h.append(mpatches.Patch(ec=mcm.RdBu(0), fc="None", label="Senkung"))
        h.append(mpatches.Patch(ec=mcm.RdBu(256), fc="None", label="Hebung"))
    else:
        h.append(mpatches.Patch(ec="k", fc="None", label="Rutschung"))
        for handle in [handle_motorway, handle_primary, handle_secondary]:
            if handle:
                h.append(handle)
    if len(h) > 0:
        ax.legend(handles=h, loc="upper right").set_zorder(10)

    # Formatting and saving
    u4ax.add_basemap(
        ax=ax, crs=crs, source=contextily.providers.CartoDB.Voyager
    )
    u4plotfmt.add_scalebar(ax=ax, width=plot_buffer * 4)
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_map.{ftype}")
        )

    plt.close(fig)


def timeseries_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    contour_path: os.PathLike,
    psi_path: os.PathLike,
    plot_buffer: float,
    overwrite: bool,
) -> Tuple[tuple, tuple]:
    """Makes a detailed overview map of the region including some geological features.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The folder where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for saving.
    :type suffix: str
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :param plot_buffer: The buffer width around the area of interest.
    :type plot_buffer: float
    """
    output_path = os.path.join(output_path, suffix)
    fexists = [
        os.path.exists(
            os.path.join(output_path, f"{row[1].group:05}_psi.{ftype}")
        )
        for ftype in IMAGE_FORMATS
    ]
    if np.all(fexists) and not overwrite:
        logging.info(f"Skipping existing plot {row[1].group:05}")
        return
    else:
        logging.info(
            f"Plotting timeseries and psi map of group {row[1].group:05}."
        )
    # Set paths
    psi_data_path = os.path.join(output_path, "psi_inv_data")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(psi_data_path, exist_ok=True)

    # Load data for plotting
    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)

    # Create figure and axes
    fig = plt.figure(figsize=(13, 5), dpi=IMAGE_DPI)
    gs = fig.add_gridspec(ncols=2, width_ratios=(1, 3))
    ax_map = fig.add_subplot(gs[0])
    ax_ts = fig.add_subplot(gs[1])

    # Plot Map
    shp_gdf.plot(ax=ax_map, fc="None", ec="C0")
    shp_gdf.buffer(plot_buffer).plot(ax=ax_map, fc="None", ec="None")

    # Load PSI data
    psi_local = u4psi.get_region_data(
        shp_gdf,
        f"grp_{row[1].group:05}_local",
        os.path.join(psi_path, "hessen_l3_clipped.gpkg"),
    )
    results = u4proc.get_psi_dict_inversion(
        psi_local,
        save_path=os.path.join(psi_data_path, f"grp_{row[1].group:05}.pkl"),
    )

    # Add PSI in Map
    ax_map.axis("equal")
    # Plot other anomalies
    if not IS_HLNUG:
        # For non HLNUG data use the thresholded contours
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax_map,
            table="thresholded_contours_all_shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    else:
        # For HLNUG data use the
        u4ax.add_gpkg_data_in_axis(
            contour_path,
            ax=ax_map,
            table="Classified_Shapes",
            edgecolor="k",
            facecolor="None",
            linewidth=1,
        )
    fig.tight_layout()
    reg_gdf = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax_map)],
        crs=crs,
    )
    psi_regional = u4psi.get_region_data(
        reg_gdf,
        f"grp_{row[1].group:05}_regional",
        os.path.join(psi_path, "hessen_l3_clipped.gpkg"),
    )
    u4spatial.xy_data_to_gdf(
        psi_regional["x"],
        psi_regional["y"],
        psi_regional["mean_vel"],
        crs=psi_regional["crs"],
    ).plot(
        ax=ax_map,
        column="data",
        cmap="RdBu",
        vmin=-5,
        vmax=5,
        zorder=2,
        markersize=30,
        edgecolors="k",
        legend=True,
        legend_kwds={
            "label": "Mean Vertical Velocity (mm/a)",
            "orientation": "horizontal",
            "shrink": 0.7,
            "extend": "both",
            "pad": 0.1,
        },
    )
    u4plotfmt.add_scalebar(ax=ax_map, width=plot_buffer * 4, div=1)
    u4ax.add_basemap(
        ax=ax_map, crs=crs, source=contextily.providers.CartoDB.Voyager
    )
    ax_map.axis("off")
    plt.draw()

    # Plot Timeseries
    u4ax.plot_timeseries_fit(
        ax=ax_ts, results=results, fit_num=1, annotate=True, show_errors=True
    )
    mylim = np.max(np.abs((ax_ts.get_ylim()))) * 1.4
    ax_ts.set_ylim(-mylim, mylim)
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Displacement (mm)")
    fig.tight_layout()
    for ftype in IMAGE_FORMATS:
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_psi.{ftype}")
        )
    plt.close(fig)


def plot_legend(
    leg_handles: list[martist.Artist],
    leg_labels: list[str],
    output_path: os.PathLike,
    group: int | str,
    suffix: str,
):
    """Creates a plot of a legend using a list of legend handles and labels.

    :param leg_handles: A list of matplotlib artists containing the symbology.
    :type leg_handles: list
    :param leg_labels: A list of labels for the artists
    :type leg_labels: list
    :param output_path: The folder where the legend is to be stored.
    :type output_path: os.PathLike
    :param group: The group number or name for the file name.
    :type group: int | str
    :param suffix: The suffix for the filename, e.g. `GK25`.
    :type suffix: str

    The size of the legend is mainly defined by the number of labels. The
    labels should be already containing newlines to fit to the width of the
    plot.

    The legend handles may overlap, for better plotting of complex map
    symbologies.
    """
    if isinstance(group, str):
        grp = group
    else:
        grp = f"{group:05}"

    logging.info("Creating legend")
    figl = plt.figure(
        figsize=(16 / 2.54, len(leg_labels) * 0.25), dpi=IMAGE_DPI
    )
    legend = figl.legend(
        handles=leg_handles, labels=leg_labels, framealpha=1, frameon=False
    )
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
    bbox = bbox.transformed(figl.dpi_scale_trans.inverted())

    for ftype in IMAGE_FORMATS:
        figl.savefig(
            os.path.join(output_path, f"{grp}_{suffix}_leg.{ftype}"),
            bbox_inches=bbox,
        )
    plt.close(figl)
