"""
Converts a folder with EGMS Data into a GPKG file.
"""

import os

import tqdm

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    project = u4projects.get_project(required=["psi_path"], interactive=False)

    file_list = u4files.get_file_list(".csv", project["paths"]["psi_path"])
    gpkg_path = os.path.join(
        project["paths"]["psi_path"], "EGMS_2018_2022.gpkg"
    )
    for file_path in tqdm(file_list, desc="Converting Files"):
        u4files.csv_to_gpkg(file_path, gpkg_path)


if __name__ == "__main__":
    main()
