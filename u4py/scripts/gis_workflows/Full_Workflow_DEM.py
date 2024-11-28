"""
**Full GIS Workflow for the reclassification of the full dataset.**

This script takes all diff DEM tiles, a set of gpkg files containing
geometries for clipping and outputs a gpkg file with contours of surface
displacements. It subdivides the study region into smaller subsets which are
processed individually, with a certain overlap.
"""

import logging
import os
from pathlib import Path

import geopandas as gp
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.files as u4files
import u4py.io.tiff as u4tiff
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    # Values for thresholding

    # Minimum area to be considered as anomaly, roughly equals a tree of
    # 17.8 m diameter.
    min_contour_area = 250

    # Levels for contouring
    contour_levels = u4spatial.plus_minus_levels([0.5])

    # Maximum distance for grouping contours
    max_contour_dist = 500

    # Options:
    use_parallel_clipping = True
    overwrite_clipping = False
    use_parallel_contouring = True
    overwrite_contours = False

    # Loading Project
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/Full_Workflow_DEM_Server.u4project"
        ).expanduser(),
        required=["base_path", "places_path", "diff_plan_path", "output_path"],
        interactive=False,
    )

    # Setting Paths
    gpkg_path = os.path.join(
        project["paths"]["places_path"], "OSM_shapes", "all_shapes.gpkg"
    )
    contour_path = os.path.join(
        project["paths"]["output_path"],
        "contours.gpkg",
    )

    # Processing
    if not os.path.exists(contour_path) or overwrite_contours:
        logging.info("Loading tiff files and clipping")
        tiff_file_list = u4files.get_file_list_tiff(
            project["paths"]["diff_plan_path"]
        )
        clipped_tiff_list = u4tiff.get_clipped_tiff_list(
            tiff_file_list,
            gpkg_path,
            overwrite=overwrite_clipping,
            use_parallel=use_parallel_clipping,
        )
        logging.info("Starting Contour extraction.")
        args = [
            (ctp, contour_levels, min_contour_area)
            for ctp in clipped_tiff_list
        ]
        if use_parallel_contouring:
            clgdf_list = u4proc.batch_mapping(
                args, u4tiff.batch_get_thresholded_contours, "Getting contours"
            )
        else:
            logging.info("Starting Single-threaded Contouring")
            clgdf_list = [
                u4tiff.batch_get_thresholded_contours(arg)
                for arg in tqdm(args, desc="Getting contours", leave=False)
            ]

        logging.info("Merging contours for geodataframe")
        data = {
            "geometry": [],
            "polygon_levels": [],
            "color_levels": [],
            "areas": [],
        }
        for clgdf in tqdm(
            clgdf_list,
            desc="Merging GDF",
            leave=False,
            total=len(clgdf_list),
        ):
            if len(clgdf) > 0:
                for k in clgdf.keys():
                    data[k].extend(clgdf[k])

        logging.info("Creating Geodataframe from results")
        gdf = gp.GeoDataFrame(
            data=data,
            crs=clgdf.crs,
        )
        gdf.to_file(contour_path)
    else:
        logging.info("Reading from Contour file")
        gdf = gp.read_file(contour_path)

    logging.info("Grouping results")
    x = gdf.centroid.x.to_numpy()
    y = gdf.centroid.y.to_numpy()
    gdf["groups"] = u4spatial.group_nearest(x, y, max_contour_dist)

    logging.info("Joining geometries of same group and level")
    gdf = u4spatial.join_groups_with_level(gdf)

    logging.info("Saving thresholded contours to disk")
    gdf.to_file(
        os.path.join(
            project["paths"]["output_path"],
            "thresholded_contours_all_shapes.gpkg",
        )
    )


if __name__ == "__main__":
    main()
