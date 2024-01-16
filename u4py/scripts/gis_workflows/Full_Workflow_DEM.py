"""
**Full GIS Workflow for the reclassification of the full dataset.**

This script takes all diff DEM tiles, a set of gpkg files containing
geometries for clipping and outputs a gpkg file with contours of surface
displacements. It subdivides the study region into smaller subsets which are
processed individually, with a certain overlap.
"""
import logging
from pathlib import Path

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj

u4config.start_logger()

import os


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/ArcGIS/U4_projects/Full_Workflow_DEM.u4project"
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
        tiff_file_list, gpkg_path, overwrite=True
    )


if __name__ == "__main__":
    main()
