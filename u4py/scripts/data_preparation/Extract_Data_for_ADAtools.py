"""
Extracts data from all datasets for use with ADAtools by Navarro et al. 2020
(https://doi.org/10.3390/ijgi9100584)
"""

import logging
import os
from pathlib import Path

import geopandas as gp
import shapely as shp

import u4py.addons.adatools as u4ada
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/Extract_Data_for_ADAtools.u4project"
        ).expanduser(),
        required=["base_path", "psi_path", "output_path"],
        interactive=False,
    )

    # First test:
    # create_test_region(project)

    # Full conversion
    gpkg_files = [
        fp
        for fp in os.listdir(project["paths"]["psi_path"])
        if fp.endswith(".gpkg")
    ]
    for gpkg_file in gpkg_files:
        logging.info(f"Converting {gpkg_file}")
        u4ada.convert_gpkg_to_shp(gpkg_file, project)
        logging.info("############")


def create_test_region(project: dict):
    """Creates example data for the test region.

    :param project: The project config
    :type project: dict
    """
    test_region = gp.GeoDataFrame(
        geometry=[
            shp.Polygon(
                [
                    (553000, 5686000),
                    (573000, 5686000),
                    (573000, 5667000),
                    (553000, 5667000),
                    (553000, 5686000),
                ]
            )
        ],
        crs="EPSG:32632",
    )
    fname = "hessen_l2b_asce_clipped.gpkg"
    tables = ["Zeitreihe_ASCE_117_07"]
    u4ada.convert_gpkg_to_shp(fname, project, tables, test_region)


if __name__ == "__main__":
    main()
