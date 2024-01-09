from datetime import timedelta
import time
from typing import Tuple
import itertools
import logging
import os
import shapely
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.sql as u4sql

u4config.start_logger()


def main():
    shp_cfg = get_shape_config()
    # project = u4proj.get_project(required=["places_path", "diff_plan_path"])
    project = dict()
    project["paths"] = {
        "base_path": "/mnt/Raid/Umwelt4/",
        "diff_plan_path": "/mnt/Raid/Umwelt4/DGM-Differenzenplan_MitKorrektur",
        "places_path": "/mnt/Raid/Umwelt4/hesse_shp",
    }

    logging.info("Calculating mask coordinates")
    tiff_file_list = u4files.get_file_list_tiff(
        project["paths"]["diff_plan_path"]
    )
    all_merged_path = os.path.join(
        project["paths"]["places_path"], "All_Merged_Shapes.gpkg"
    )
    if os.path.exists(all_merged_path):
        logging.info("Loading buffered shapes from file.")
        merged_gdf = gp.GeoDataFrame.from_file(all_merged_path)
    else:
        logging.info("Recreating new buffered shapes")
        merged_gdf = tile_shapefiles(
            all_merged_path, tiff_file_list, project, shp_cfg
        )

    with Pool(u4config.cpu_count) as pool:
        logging.info("Batch clipping tiffs")
        tiffout_folder = os.path.join(
            project["paths"]["diff_plan_path"], "clipped_tiffs"
        )
        os.makedirs(tiffout_folder, exist_ok=True)
        # pool.map(
        #     u4files.batch_clip_tiff,
        #     zip(
        #         tiff_file_list,
        #         itertools.repeat(tiffout_folder),
        #         itertools.repeat(merged_gdf),
        #     ),
        # )
        for tf in tqdm(tiff_file_list):
            u4files.clip_tiff(tf, tiffout_folder, merged_gdf)


def tile_shapefiles(all_merged_path, tiff_file_list, project, shp_cfg):
    logging.info("Starting parallel Pool for processing")
    with Pool(u4config.cpu_count) as pool:
        logging.info("Starting to get buffered data")
        buff_geom = []
        for kk in tqdm(
            shp_cfg["shp_file"].keys(), desc="Reading and buffering files"
        ):
            buff_geom.extend(
                u4sql.read_buf(
                    project["paths"]["places_path"], shp_cfg, kk, pool=pool
                )
            )
        # buff_geom = u4sql.read_buf(
        #     project["paths"]["places_path"], shp_cfg, "build"
        # )

        minx_hess = 409000
        miny_hess = 5470500
        maxx_hess = 589000
        maxy_hess = 5724500

        geom_sort, outofbounds = sort_geometry(
            buff_geom, minx_hess, miny_hess, maxx_hess, maxy_hess, pool=pool
        )
        out_folder = os.path.join(
            project["paths"]["places_path"], "buffmerged_tiled_shps"
        )
        os.makedirs(out_folder, exist_ok=True)
        out_paths = [
            os.path.join(out_folder, f"buffmerged_tile{ii:04}.gpkg")
            for ii in range(len(geom_sort))
        ]

        logging.info("Saving files sorted geometries")
        pool.map(unify_worker, zip(geom_sort, out_paths))

        logging.info("Merging gpkgs")
        gpkg_paths = [
            os.path.join(out_folder, f)
            for f in os.listdir(out_folder)
            if f.endswith(".gpkg")
        ]
        data_for_merging = pool.map(merging_worker, gpkg_paths)
        logging.info("Creating merged GDF")
        merged_gdf = gp.GeoDataFrame(
            geometry=data_for_merging, crs="EPSG:32632"
        )

        logging.info("Saving merged GDF")
        merged_gdf.to_file(all_merged_path)
    return merged_gdf


def merging_worker(gpkg_path):
    return gp.GeoSeries.from_file(gpkg_path).unary_union


def unify_worker(args):
    gs = gp.GeoSeries(args[0], crs="EPSG:32632")
    gdf = gp.GeoDataFrame(geometry=[gs.unary_union], crs="EPSG:32632")
    gdf.to_file(args[1])


def sort_geometry(
    geometries: list,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    step: float = 10000,
    pool=False,
) -> Tuple[list, list]:
    logging.info("Getting sorting indices in parallel")
    if not pool:
        soolort_indices = p.map(
            sort_indices_worker,
            zip(
                geometries,
                itertools.repeat(minx),
                itertools.repeat(miny),
                itertools.repeat(step),
            ),
        )
    else:
        sort_indices = pool.map(
            sort_indices_worker,
            zip(
                geometries,
                itertools.repeat(minx),
                itertools.repeat(miny),
                itertools.repeat(step),
            ),
        )

    logging.info("Sorting data in boxes")
    num_xboxes = int(np.ceil((maxx - minx) / step)) + 1
    num_yboxes = int(np.ceil((maxy - miny) / step)) + 1
    box_list = [[[] for i in range(num_yboxes)] for j in range(num_xboxes)]
    outofbounds = []
    for g, (x, y) in zip(geometries, sort_indices):
        if x <= num_xboxes and y <= num_yboxes and x >= 0 and y >= 0:
            box_list[x][y].append(g)
        else:
            outofbounds.append(g)
    all_list = []
    for g in box_list:
        for a in g:
            if a:
                all_list.append(a)
    return all_list, outofbounds


def sort_indices_worker(args):
    geometry, minx, miny, step = args
    centroid = shapely.centroid(geometry)
    ix = int(np.floor((centroid.x - minx) / step))
    iy = int(np.floor((centroid.y - miny) / step))
    return (ix, iy)


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
        "build": "gis_osm_buildings_a_free_1.gpkg",
        "landuse": "gis_osm_landuse_a_free_1.gpkg",
        "construction": "lan-con.gpkg",
        "railway": "gis_osm_railways_free_1.gpkg",
        "mainroads": "gis_osm_roads_free_1.gpkg",
        "minor_roads": "gis_osm_roads_free_1.gpkg",
        "water": "gis_osm_waterways_free_1.gpkg",
        "lakes": "gis_osm_water_a_free_1.gpkg",
        "power": "pow-gen_gen-win.gpkg",
        "parking": "par-sur.gpkg",
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
    tic = time.time()
    main()
    elapsed = timedelta(seconds=time.time() - tic)
    logging.info(f"Finished after {elapsed}.")
