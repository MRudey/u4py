import itertools
import logging
import os
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import Tuple

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import shapely
from tqdm import tqdm

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.sql as u4sql

u4config.start_logger()


def main():
    shp_cfg = u4config.get_shape_config()
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


if __name__ == "__main__":
    tic = time.time()
    main()
    elapsed = timedelta(seconds=time.time() - tic)
    logging.info(f"Finished after {elapsed}.")
