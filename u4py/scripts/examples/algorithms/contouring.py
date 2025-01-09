"""
This script shows how the contouring of the difference plans with the OSM data
works.
"""

import logging
import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np

import u4py.analysis.spatial as u4spatial
import u4py.io.gpkg as u4gpkg
import u4py.io.sql as u4sql
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4pltfmt
import u4py.utils.config as u4config

u4config.start_logger()


def main():
    base_path = Path(
        r"~\Documents\ArcGIS\EndberichtWorkflowPlots"
    ).expanduser()
    output_path = Path(
        r"~\HESSENBOX-DA\Umwelt_4\Bericht_2024\images"
    ).expanduser()

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True
    )

    # Original difference plan and boundary of plot
    logging.info("Plotting Differenzenplan")
    axes[0][0].set_title("Differenzenplan")
    orig_folder = os.path.join(base_path, "original_tiffs")
    orig_tiffs = [
        os.path.join(orig_folder, f)
        for f in os.listdir(orig_folder)
        if f.endswith(".tif")
    ]
    bounds = []
    for tile in orig_tiffs:
        bds, crs = u4ax.add_tile(tile, ax=axes[0][0], vm=2, cmap="RdBu")
        bounds.append(bds)
    xmin = np.min([bds.left for bds in bounds])
    xmax = np.max([bds.right for bds in bounds])
    ymin = np.min([bds.bottom for bds in bounds])
    ymax = np.max([bds.top for bds in bounds])

    axes[0][0].set_xlim(xmin, xmax)
    axes[0][0].set_ylim(ymin, ymax)

    # region_gdf = gp.GeoDataFrame(
    #     geometry=[u4spatial.bounds_to_polygon((xmin, ymin, xmax, ymax))],
    #     crs=crs,
    # )

    # Get and plot buffered shapes
    logging.info("Plotting Filter Masken")
    axes[0][1].set_title("Filter Masken")
    shp_cfg = u4config.get_shape_config()
    shape_cache_folder = os.path.join(base_path, "shape_cache")
    os.makedirs(shape_cache_folder, exist_ok=True)
    for tile in orig_tiffs:
        fname = os.path.splitext(os.path.split(tile)[-1])[0]
        shppath = os.path.join(shape_cache_folder, fname + ".shp")
        if os.path.exists(shppath):
            bfshp = gp.read_file(shppath)
        else:
            bfshp = u4gpkg.load_and_buffer_gpkg(
                os.path.join(base_path, "all_shapes.gpkg"),
                tile,
                shp_cfg=shp_cfg,
            )
            bfshp.to_file(shppath)
        bfshp.plot(ax=axes[0][1], color="k")

    # Plot Clipped tiffs
    logging.info("Plotting Clipped Tiffs")
    axes[0][2].set_title("Geschnittener Differenzenplan")
    clip_folder = os.path.join(base_path, "clipped_tiffs")
    clip_tiffs = [
        os.path.join(clip_folder, f)
        for f in os.listdir(clip_folder)
        if f.endswith(".tif")
    ]
    for tile in clip_tiffs:
        u4ax.add_tile(tile, ax=axes[0][2], vm=2, cmap="RdBu")

    # Contours
    logging.info("Plotting Contours")
    axes[1][0].set_title("Konturierung")
    u4ax.add_gpkg_data_in_axis(
        os.path.join(base_path, "contours.gpkg"),
        table="contours",
        ax=axes[1][0],
        ax_crs=crs,
        column="color_levels",
        fc="None",
        cmap="RdBu",
    )

    # Groups
    logging.info("Plotting Groups")
    axes[1][1].set_title("Gruppierung")
    u4ax.add_gpkg_data_in_axis(
        os.path.join(base_path, "Classified_Shapes.gpkg"),
        table="Classified_Shapes",
        ax=axes[1][1],
        ax_crs=crs,
        column="group",
        fc="None",
        cmap="tab10",
    )

    # Klassifikation
    logging.info("Plotting Classification")
    axes[1][2].set_title("Klassifikation")
    gpkg_crs = u4sql.get_crs(
        os.path.join(base_path, "Filtered_Classified_Shapes_onlyLarge.gpkg"),
        "Filtered_Classified_Shapes_2410",
    )[0]
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(axes[1][2])], crs=crs
    ).to_crs(gpkg_crs)
    data = u4gpkg.load_gpkg_data_region_ogr(
        region,
        os.path.join(base_path, "Filtered_Classified_Shapes_onlyLarge.gpkg"),
        "Filtered_Classified_Shapes_2410",
        clip=False,
    )
    data.plot(ax=axes[1][2], column="manual_class_1", fc="None", ec="C1")
    data.apply(
        lambda x: axes[1][2].annotate(
            text=f"{x['manual_class_1']}\n{x['manual_class_2']}\n{x['manual_class_3']}",
            xy=x.geometry.centroid.coords[0],
            xytext=(1, 1),
            textcoords="offset fontsize",
            ha="center",
            va="center",
        ),
        axis=1,
    )

    # Format and save
    logging.info("Formatting and Saving")
    u4ax.add_basemap(ax=axes[0][1], crs=crs)
    u4ax.add_basemap(ax=axes[1][0], crs=crs)
    u4ax.add_basemap(ax=axes[1][1], crs=crs)
    u4ax.add_basemap(ax=axes[1][2], crs=crs)
    fig.tight_layout()
    axes[0][0].yaxis.set_major_formatter(u4pltfmt.coordinate_formatter)
    axes[1][0].yaxis.set_major_formatter(u4pltfmt.coordinate_formatter)
    axes[1][0].xaxis.set_major_formatter(u4pltfmt.coordinate_formatter)
    axes[1][1].xaxis.set_major_formatter(u4pltfmt.coordinate_formatter)
    axes[1][2].xaxis.set_major_formatter(u4pltfmt.coordinate_formatter)
    fig.savefig(os.path.join(output_path, "ContourWorkflow.png"))
    fig.savefig(os.path.join(output_path, "ContourWorkflow.pdf"))


if __name__ == "__main__":
    main()
