"""
Clips the Tiffs in a given region by the merged shape file and merges the
results into a single large tiff. This can be thresholded later with
`Find_Regions_in_Merged_Tif.py`
"""


import logging
import os
from pathlib import Path

import contextily
import geopandas as gp

# import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.plot as rioplot
from tqdm import tqdm

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    # Paths are stored in project file
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/Clip_and_Merge_DEM_Tiles_Server.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "places_path",
            "piloten_path",
            "diff_plan_path",
        ],
        interactive=False,
    )

    rois = gp.read_file(project["paths"]["piloten_path"])
    shp_cfg = u4config.get_shape_config()
    overwrite = True
    excluded_regions = []
    for region_name in tqdm(
        rois.Name,
        desc="Clipping and Merging Tiffs",
        total=len(rois.geometry),
    ):
        if not region_name in excluded_regions:
            logging.info(f"Starting with region: {region_name}")
            roi = rois[rois.Name == region_name]
            region_name = region_name.replace(" ", "")
            tiff_file_list = u4files.get_region_tiff(
                roi, region_name, project["paths"]["diff_plan_path"]
            )
            if len(tiff_file_list) > 1000:
                logging.info("Region too large, subdividing.")
                roi_file_list, new_rois = u4files.get_subdivided_roi_file_list(
                    roi, region_name, project["paths"]["diff_plan_path"]
                )
                for ii in tqdm(
                    range(len(roi_file_list)),
                    desc="Processing subdivisions",
                    leave=False,
                ):
                    clip_merge_region(
                        roi_file_list[ii],
                        project,
                        new_rois[ii],
                        region_name + f"_{ii}",
                        shp_cfg,
                        overwrite=overwrite,
                    )
            else:
                clip_merge_region(
                    tiff_file_list,
                    project,
                    roi,
                    region_name,
                    shp_cfg,
                    overwrite=overwrite,
                )

        else:
            logging.info(f"{region_name} on ignore list.")

    # fig, ax = plt.subplots()
    # with rio.open(merged_tiff_file_path, "r") as tile:
    #     rioplot.show(tile, ax=ax, cmap="RdBu", vmin=-2, vmax=2)
    # # _show_selected_tiffs(clipped_tiffs_list, ax=ax)
    # # _show_merged_data(merged_data, ax=ax)
    # plt.show()


def clip_merge_region(
    tiff_file_list, project, roi, region_name, shp_cfg, overwrite
):
    with rio.open(tiff_file_list[0], "r") as tile:
        tiff_crs = tile.crs.to_string()
    merged_data, merged_path = u4files.get_buffered_shapefiles(
        project["paths"]["places_path"],
        roi,
        region_name,
        shp_cfg,
        out_crs=tiff_crs,
        overwrite=overwrite,
    )

    clipped_tiffs_list = u4files.get_clipped_tiff_list(
        tiff_file_list,
        merged_path,
        region_name=region_name,
        overwrite=overwrite,
    )

    clipped_tiff_folder, _ = os.path.split(clipped_tiffs_list[0])
    merged_tiff_file_path = u4files.get_merged_tiff_path(
        clipped_tiff_folder, mask=roi, overwrite=overwrite
    )


def _show_merged_data(merged_data, ax):
    merged_data.plot(ax=ax, zorder=2)


def _show_shp_data(shp_data, shp_cfg, ax):
    for kk in shp_data.keys():
        shp_data[kk].plot(
            ax=ax, color=shp_cfg["colors"][kk], zorder=shp_cfg["zorder"][kk]
        )


def _show_selected_tiffs(tiff_file_list, ax):
    x = []
    y = []
    for tf in tqdm(tiff_file_list, desc="Reading Rasters"):
        tile = u4files.load_tiff(tf)
        x.extend([tile.bounds[0], tile.bounds[2]])
        y.extend([tile.bounds[1], tile.bounds[3]])
        rioplot.show(tile, ax=ax, cmap="RdBu", vmin=-2, vmax=2)
        tile.close()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))


def _show_points_map(points, crs, ax):
    points.plot(ax=ax)
    contextily.add_basemap(
        ax,
        crs=crs,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )


if __name__ == "__main__":
    main()
