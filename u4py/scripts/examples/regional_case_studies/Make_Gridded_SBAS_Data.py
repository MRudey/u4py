"""Compares the SBAS results and BBD results in several regions"""

# import pickle as pkl
import logging
import os
import shutil
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.gpkg as u4gpkg
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff


def main():
    use_parallel = True
    dx = 25
    width = 2500

    sbas_folder = "/mnt/Raid/Umwelt4/SBAS"
    # sbas_folder = Path(r"~\Documents\ArcGIS\SBAS").expanduser()
    bbd_path = "/mnt/Raid/Umwelt4/hessen_L2B/hessen_l2b_desc_clipped.gpkg"
    # bbd_path = Path(
    #     r"~\Documents\ArcGIS\Data_2023\hessen_L2B\hessen_l2b_desc_clipped.gpkg"
    # ).expanduser()

    file_list = [
        os.path.join(sbas_folder, fp)
        for fp in os.listdir(sbas_folder)
        if fp.endswith(".gpkg")
    ]

    for sbas_path in file_list:
        grid_data(bbd_path, sbas_path, dx, width, use_parallel)


def grid_data(
    bbd_path: os.PathLike | str,
    sbas_path: os.PathLike | str,
    dx: float,
    width: float,
    use_parallel: bool,
):
    if (width % dx) != 0:
        ValueError("Width is not a multiple of dx.")

    table = os.path.splitext(os.path.split(sbas_path)[-1])[0]
    folder = os.path.split(sbas_path)[0]
    tiff_folder = os.path.join(folder, f"{table}_tiffs")
    if os.path.exists(tiff_folder):
        shutil.rmtree(tiff_folder)
    os.makedirs(tiff_folder, exist_ok=True)

    bounds = u4sql.get_bounds(sbas_path)[0]
    bounds_sbas = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(bounds)],
        crs=u4sql.get_crs(sbas_path, table)[0],
    )
    subd_sbas = u4spatial.subdivide_gdf_by_squares(bounds_sbas, width=width)
    subd_bbd = subd_sbas.to_crs(
        u4sql.get_crs(bbd_path, "mittlere_Geschwindigkeit")[0]
    )

    args = [
        (reg_sbas, sbas_path, reg_bbd, bbd_path, table, dx, ii)
        for ii, (reg_sbas, reg_bbd) in enumerate(
            zip(subd_sbas.geometry, subd_bbd.geometry)
        )
    ]
    if use_parallel:
        u4proc.batch_mapping(args, rectify_wrapper, "Processing layers")
    else:
        for arg in tqdm(args):
            rectify_wrapper(arg)


def rectify_wrapper(args):
    rectify(*args)


def rectify(reg_sbas, sbas_path, reg_bbd, bbd_path, table, dx, ii):
    folder = os.path.split(sbas_path)[0]
    tiff_folder = os.path.join(folder, f"{table}_tiffs")
    sbas_folder = os.path.join(tiff_folder, "sbas")
    bbd_folder = os.path.join(tiff_folder, "bbd")
    os.makedirs(sbas_folder, exist_ok=True)
    os.makedirs(bbd_folder, exist_ok=True)

    logging.info("Loading sbas_data")
    sbas_data = u4gpkg.load_gpkg_data_region_ogr(reg_sbas, sbas_path, table)
    logging.info("Loading bbd_data")
    bbd_data = u4gpkg.load_gpkg_data_region_ogr(
        reg_bbd, bbd_path, "mittlere_Geschwindigkeit"
    )
    bounds = reg_bbd.bounds
    xq = np.arange(bounds[0], bounds[2], dx)
    yq = np.arange(bounds[1], bounds[3], dx)

    if len(sbas_data) > 0:
        logging.info("Rectifying vv_sbas data.")
        vv_sbas = rectify_data(
            xq,
            yq,
            sbas_data.velocity.to_numpy(),
            sbas_data.geometry.x.to_numpy(),
            sbas_data.geometry.y.to_numpy(),
            dx,
        )
    else:
        logging.info("No data found for vv_sbas")
        vv_sbas = np.ones((len(yq), len(xq))) * np.nan

    logging.info("Saving vv_sbas to geotiff.")
    u4tiff.ndarray_to_geotiff(
        vv_sbas,
        [bounds[0], bounds[2], bounds[1], bounds[3]],
        os.path.join(sbas_folder, f"{table}_tile_{ii}.tiff"),
        crs=u4sql.get_crs(bbd_path, "mittlere_Geschwindigkeit")[0],
    )

    if len(bbd_data) > 0:
        logging.info("Rectifying vv_bbd data.")
        vv_bbd = rectify_data(
            xq,
            yq,
            bbd_data.mean_velocity.to_numpy(),
            bbd_data.geometry.x.to_numpy(),
            bbd_data.geometry.y.to_numpy(),
            dx,
        )
    else:
        logging.info("No data found for vv_bbd")
        vv_bbd = np.ones((len(yq), len(xq))) * np.nan

    logging.info("Saving vv_bbd to geotiff.")
    u4tiff.ndarray_to_geotiff(
        vv_bbd,
        [bounds[0], bounds[2], bounds[1], bounds[3]],
        os.path.join(bbd_folder, f"{table}_tile_{ii}.tiff"),
        crs=u4sql.get_crs(bbd_path, "mittlere_Geschwindigkeit")[0],
    )


def rectify_data(
    xq: np.ndarray,
    yq: np.ndarray,
    vv_data: np.ndarray,
    xx_data: np.ndarray,
    yy_data: np.ndarray,
    dx: float,
):
    vv = np.ones((len(yq), len(xq))) * np.nan
    for ii, x in enumerate(xq):
        x_slc = np.logical_and(xx_data <= x + dx / 2, xx_data >= x - dx / 2)
        for jj, y in enumerate(yq):
            y_slc = np.logical_and(
                yy_data <= y + dx / 2, yy_data >= y - dx / 2
            )
            slc = np.logical_and(x_slc, y_slc)
            if len(vv_data[slc] > 0):
                vv[jj, ii] = np.nanmean(vv_data[slc])
    return vv


def plot_gridded_data(
    xq: np.ndarray, yq: np.ndarray, bbd_vv: np.ndarray, sbas_vv: np.ndarray
):
    fig, axes = plt.subplots(
        ncols=3, figsize=(15, 7), sharex=True, sharey=True
    )
    axes[0].imshow(
        bbd_vv,
        origin="lower",
        extent=(xq[0], xq[-1], yq[0], yq[-1]),
        vmin=-20,
        vmax=20,
        cmap="RdYlBu",
        zorder=1,
    )
    axes[1].imshow(
        sbas_vv,
        origin="lower",
        extent=(xq[0], xq[-1], yq[0], yq[-1]),
        vmin=-20,
        vmax=20,
        cmap="RdYlBu",
        zorder=1,
    )
    axes[2].imshow(
        bbd_vv - sbas_vv,
        origin="lower",
        extent=(xq[0], xq[-1], yq[0], yq[-1]),
        vmin=-20,
        vmax=20,
        cmap="RdYlBu",
        zorder=1,
    )


def plot_raw_data(bbd_data: gp.GeoDataFrame, sbas_data: gp.GeoDataFrame):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    bbd_data.plot(
        ax=axes[0],
        column="mean_velocity",
        marker="s",
        colormap="RdYlBu",
        vmin=-20,
        vmax=20,
    )
    sbas_data.plot(
        ax=axes[1],
        column="velocity",
        marker=".",
        colormap="RdYlBu",
        vmin=-20,
        vmax=20,
    )


if __name__ == "__main__":
    main()
