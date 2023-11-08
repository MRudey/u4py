"""
Reclassifies the data in a given test region.

Test-Region Calculations

    - Import
    - Data preparation
    - Calculations

Converted from: `220423_template_aoi.ipynb`
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
            "~/Documents/umwelt4/Reclassify_GroundMotions_Server.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "places_path",
            "piloten_path",
            "diff_plan_path",
        ],
    )

    rois = gp.read_file(project["paths"]["piloten_path"])
    shp_cfg = get_shape_config()
    overwrite = True
    excluded_regions = ["Rhein-Main"]
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
        else:
            logging.info(f"{region_name} on ignore list.")

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
        "landuse": "Steinbruch & Industrie",
        "construction": "Baustellen",
        "railway": "Railways",
        "mainroads": "Main Roads",
        "minor_roads": "Kleine Straßen",
        "water": "Waterways",
        "lakes": "Lakes",
        "power": "Windkraftanlagen",
        "parking": "Parkplätze",
    }
    # Name of the shapefile containing the original data
    shp_file = {
        "build": "gis_osm_buildings_a_free_1.shp",
        "landuse": "gis_osm_landuse_a_free_1.shp",
        "construction": "lan-con.shp",
        "railway": "gis_osm_railways_free_1.shp",
        "mainroads": "gis_osm_roads_free_1.shp",
        "minor_roads": "gis_osm_roads_free_1.shp",
        "water": "gis_osm_waterways_free_1.shp",
        "lakes": "gis_osm_water_a_free_1.shp",
        "power": "pow-gen_gen-win.shp",
        "parking": "par-sur.shp",
    }
    # List of feature classes to extract from the file (empty=all)
    fclass = {
        "build": [],
        "landuse": ["quarry", "construction", "industrial"],
        "construction": [],
        "railway": [],
        "mainroads": [
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
            "unclassified",  # necessary for some slip roads
        ],
        "minor_roads": [
            "residential",
            "living_street",
            "service",
            "pedestrian",
            "track",
            "track_grade5",
            "track_grade4",
            "track_grade3",
            "track_grade2",
            "track_grade1",
            "road",
            "bridleway",
            "steps",
            "path",
            "cycleway",
        ],
        "water": [],
        "lakes": [],
        "power": [],
        "parking": [],
    }
    # Default buffer size around each feature is 10 meters.
    buffer_dist = {
        "build": 10,
        "landuse": 10,
        "construction": 10,
        "railway": 15,
        "mainroads": 15,
        "minor_roads": 5,
        "water": 10,
        "lakes": 10,
        "power": 150,
        "parking": 10,
    }

    # Plotting Stuff
    colors = {
        "build": "dimgray",
        "landuse": "rosybrown",
        "construction": "rosybrown",
        "railway": "black",
        "mainroads": "orange",
        "minor_roads": "green",
        "water": "aqua",
        "lakes": "aqua",
        "power": "yellow",
        "parking": "blue",
    }
    zorder = {
        "build": 2,
        "landuse": 1,
        "construction": 1,
        "railway": 5,
        "mainroads": 4,
        "minor_roads": 3,
        "water": 3,
        "lakes": 3,
        "power": 3,
        "parking": 3,
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
