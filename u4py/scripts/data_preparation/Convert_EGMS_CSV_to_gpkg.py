"""
Converts a folder with EGMS Data into a GPKG file.
"""

import logging
import os

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    project = u4projects.get_project(
        proj_path="/home/rudolf/Documents/umwelt4/ConvertEGMS.u4project",
        required=["psi_path"],
        interactive=True,
    )

    file_list = u4files.get_file_list(".csv", project["paths"]["psi_path"])
    gpkg_path = os.path.join(project["paths"]["psi_path"], "EGMS_2015_2021")
    merged_v, merged_ew = u4files.egms_csv_list_to_gdf(file_list)
    logging.info("Saving vertical data")
    merged_v.to_file(gpkg_path + "_vertikal.gpkg")
    logging.info("Saving East-West data")
    merged_ew.to_file(gpkg_path + "_ost_west.gpkg")
    logging.info("Done!")


if __name__ == "__main__":
    main()
