"""
Plots all PSI from the gpkg file into geotiffs and saves them in a folder.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        required=["base_path"],
        proj_path=Path(
            "~/Documents/umwelt4/Convert_EGMS_gpkg_to_tif.u4project"
        ).expanduser(),
    )
    folder_list = [
        "Data_EGMS_2015_2021",
        "Data_EGMS_2018_2022",
        "Data_EGMS_2019_2023",
    ]
    file_list = []
    for fp in folder_list:
        folder_path = os.path.join(project["paths"]["base_path"], fp)
        file_list.extend(
            [
                os.path.join(folder_path, gpkg_fp)
                for gpkg_fp in os.listdir(folder_path)
                if gpkg_fp.endswith(".gpkg")
            ]
        )

    for fp in tqdm(file_list, desc="Converting gpkg files", leave=False):
        egms_gpkg_to_tiff(fp, project["paths"]["base_path"])


def egms_gpkg_to_tiff(
    gpkg_file_path: os.PathLike | str, output_path: os.PathLike | str
):
    """
    Converts the timeseries data stored in the egms gpkg file to a set of
    geotiffs for faster display. Extracts data from all directions in the
    given file.

    :param gpkg_file_path: The path to the gpkg file.
    :type gpkg_file_path: os.PathLike | str
    :param output_path: The path where the folder with the tiffs shall be stored.
    :type output_path: os.PathLike | str
    """
    directions = u4sql.get_table_names(gpkg_file_path)
    for direction in tqdm(
        directions, desc="Extracting directions", leave=False
    ):
        # Setting up paths
        output_dir = os.path.join(output_path, f"PSI_Tiffs_{direction}")
        os.makedirs(output_dir, exist_ok=True)

        # Reading common data, elevation data (zs) not used atm
        xs, ys, _, time, queries, info = u4sql.gen_queries_psi_gpkg(
            gpkg_file_path,
            direction,
        )
        spacing = np.min(
            [
                np.min(np.diff(np.sort(np.unique(xs)))),
                np.min(np.diff(np.sort(np.unique(ys)))),
            ]
        )
        xq = np.array((xs - np.min(xs)) / spacing, dtype=int)
        yq = np.array((ys - np.min(ys)) / spacing, dtype=int)
        extent = (  # Adjustments necessary to match up coordinates
            np.min(xs) + spacing / 2,
            np.max(xs) + spacing * 1.5,
            np.min(ys) - spacing / 2,
            np.max(ys) + spacing / 2,
        )

        # Build arguments list (for easier parallelization, WIP)
        arguments = [
            (xq, yq, time, extent, output_dir, info, q) for q in queries
        ]
        u4proc.batch_mapping(
            arguments, multi_geotiff, desc="Generating GeoTiffs"
        )


def multi_geotiff(arguments: tuple):
    """Takes the values in the tuple and creates a geotiff.

    The arguments contain data that is the same for all geotiffs and a sql
    query that is used to read the data from the input file.

    :param arguments: The arguments for plotting
    :type arguments: tuple
    """
    # Disassemble arguments
    xq, yq, time, extent, output_dir, info, query = arguments

    # Read data and fit them into velocity array
    data = u4sql.single_query(*query)
    vels = np.ones((np.max(yq) + 1, np.max(xq) + 1)) * np.nan
    vel = np.array([d if d else np.nan for d in data[0]])
    vels[yq, xq] = vel

    # Read timestamp
    time_str = datetime.strftime(time[query[2]], "%Y-%m-%d")
    fname = f"{time_str}.tif"

    # Create GeoTiff
    u4tiff.ndarray_to_geotiff(
        vels,
        extent,
        os.path.join(output_dir, fname),
        crs="EPSG:3035",
        compress="lzw",
    )


if __name__ == "__main__":
    main()
