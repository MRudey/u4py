"""
**Full GIS Workflow for the reclassification of the full dataset.**

This script takes all diff DEM tiles, a set of gpkg files containing
geometries for clipping and outputs a gpkg file with contours of surface
displacements. It subdivides the study region into smaller subsets which are
processed individually, with a certain overlap.
"""
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import geopandas as gp
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/Full_Workflow_DEM_Server.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "places_path",
            "diff_plan_path",
        ],
        interactive=False,
    )

    tiff_file_list = u4files.get_file_list_tiff(
        project["paths"]["diff_plan_path"]
    )
    gpkg_path = os.path.join(
        project["paths"]["places_path"], "OSM_shapes", "all_shapes.gpkg"
    )

    clipped_tiff_list = u4files.get_clipped_tiff_list_gpkg(
        tiff_file_list, gpkg_path, overwrite=False, use_parallel=True
    )
    logging.info("Starting Contour extraction.")
    threshold = 250  # => Tree with approx. 17.8 m diameter
    levels = u4spatial.plus_minus_levels([0.5, 1, 2, 5, 10])

    args = [(ctp, levels, threshold) for ctp in clipped_tiff_list]
    # Parallel
    with Pool(u4config.cpu_count) as p:
        logging.info("Starting Parallel Pool")
        clgdf_list = list(
            tqdm(
                p.imap_unordered(batch_get_thresholded_contours, args),
                total=len(tiff_file_list),
                desc="Getting contours",
                leave=False,
            )
        )

    # Non Parallel
    # clgdf_list = [batch_get_thresholded_contours(arg) for arg in args]

    data = create_empty_dict(clgdf_list[0])
    for clgdf in tqdm(clgdf_list, desc="Merging GDF"):
        for k in clgdf.keys():
            data[k].extend(clgdf[k])
    logging.info("Creating Geodataframe from results")
    gdf = gp.GeoDataFrame(
        data=data,
        crs=clgdf.crs,
    )
    logging.info("Saving thresholded contours to disk")
    gdf.to_file(
        os.path.join(
            project["paths"]["places_path"],
            "thresholded_contours_all_shapes.gpkg",
        )
    )


def batch_get_thresholded_contours(args):
    gdf = u4files.get_thresholded_contours(*args, save_intermediate=False)
    return gdf


def create_empty_dict(gdf):
    data = dict()
    for k in gdf.keys():
        data[k] = []
    return data


if __name__ == "__main__":
    main()
