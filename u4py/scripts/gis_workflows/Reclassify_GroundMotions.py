"""
Reclassifies the data in a given test region.

Test-Region Calculations

    - Import
    - Data preparation
    - Calculations

Converted from: `220423_template_aoi.ipynb`
"""


import os

import contextily
import geopandas as gp
import matplotlib.pyplot as plt
import rasterio as rio
import rasterio.plot as rioplot
from tqdm import tqdm

import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    # Paths are stored in project file
    project = u4proj.get_project(
        required=["base_path", "places_path", "piloten_path", "diff_plan_path"]
    )

    rois = gp.read_file(project["paths"]["piloten_path"])
    shp_cfg = get_shape_config()
    for region_name in tqdm(
        rois.Name,
        desc="Clipping and Merging Tiffs",
        total=len(rois.geometry),
    ):
        roi = rois[rois.Name == region_name]
        region_name = region_name.replace(" ", "")
        tiff_file_list = u4files.get_region_tiff(
            roi, region_name, project["paths"]["diff_plan_path"]
        )
        with rio.open(tiff_file_list[0], "r") as tile:
            tiff_crs = tile.crs.to_string()
        merged_data, merged_path = u4files.get_buffered_shapefiles(
            project["paths"]["places_path"],
            roi,
            region_name,
            shp_cfg,
            out_crs=tiff_crs,
        )

        clipped_tiffs_list = u4files.get_clipped_tiff_list(
            tiff_file_list, merged_path, region_name=region_name
        )

        clipped_tiff_folder, _ = os.path.split(clipped_tiffs_list[0])
        merged_tiff_file_path = u4files.get_merged_tiff_path(
            clipped_tiff_folder, mask=roi
        )

    # fig, ax = plt.subplots()
    # with rio.open(merged_tiff_file_path, "r") as tile:
    #     rioplot.show(tile, ax=ax, cmap="RdBu", vmin=-2, vmax=2)
    # # _show_selected_tiffs(clipped_tiffs_list, ax=ax)
    # # _show_merged_data(merged_data, ax=ax)
    # plt.show()


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


def get_shape_config():
    # Names of the shapes for the legend
    name = {
        "build": "Buildings",
        "quarry": "Quarries",
        "railway": "Railways",
        "roads": "Roads",
        "water": "Waterways",
        "lakes": "Lakes",
    }
    # Name of the shapefile containing the original data
    shp_file = {
        "build": "gis_osm_buildings_a_free_1.shp",
        "quarry": "gis_osm_landuse_a_free_1.shp",
        "railway": "gis_osm_railways_free_1.shp",
        "roads": "gis_osm_roads_free_1.shp",
        "water": "gis_osm_waterways_free_1.shp",
        "lakes": "gis_osm_water_a_free_1.shp",
    }
    # List of feature classes to extract from the file (empty=all)
    fclass = {
        "build": [],
        "quarry": ["quarry"],
        "railway": [],
        "roads": [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
        ],
        "water": [],
        "lakes": [],
    }
    # Default buffer size around each feature is 10 meters.
    buffer_dist = {
        "build": 10,
        "quarry": 10,
        "railway": 10,
        "roads": 10,
        "water": 10,
        "lakes": 10,
    }

    # Plotting Stuff
    colors = {
        "build": "dimgray",
        "quarry": "rosybrown",
        "railway": "black",
        "roads": "orange",
        "water": "aqua",
        "lakes": "aqua",
    }
    zorder = {
        "build": 2,
        "quarry": 1,
        "railway": 5,
        "roads": 4,
        "water": 3,
        "lakes": 3,
    }

    shp_cfg = {
        "buffer_dist": buffer_dist,
        "name": name,
        "shp_file": shp_file,
        "fclass": fclass,
        "colors": colors,
        "zorder": zorder,
    }

    return shp_cfg


if __name__ == "__main__":
    main()
