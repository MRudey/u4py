"""
Submodule for handling csv and other text based data formats
"""

import csv
import logging
import os
from typing import Tuple

import geopandas as gp
import numpy as np
import shapely
from tqdm import tqdm


def load_csv(file_path: os.PathLike) -> dict:
    """Loads a CSV File from the EGMS dataset

    :param file_path: The path to file.
    :type file_path: os.PathLike
    :return: A dictionary with the data and the headers as keys.
    :rtype: dict
    """
    num_lines = buf_count_newlines_gen(file_path) - 1
    logging.info("Loading CSV file")
    with open(file_path, "rt", encoding="utf-8", newline="\n") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

        data = dict()
        for ii, k in enumerate(header):
            if ii > 0:
                data[k] = np.ones((num_lines,))
                # if k in ["easting", "northing", "height"]:
                #     data[k] = np.ones((num_lines,))
                # else:
                #     pass
            else:
                data[k] = []
        for irow, row in enumerate(reader):
            for ii, (v, k) in enumerate(zip(row, header)):
                if ii > 0:
                    data[k][irow] = float(v)
                    # if k in ["easting", "northing", "height"]:
                    #     data[k][irow] = float(v)
                    # else:
                    #     pass
                else:
                    data[k].append(v)
    logging.info("Creating geometries")
    data["geometry"] = [
        shapely.Point(x, y, z)
        for x, y, z in zip(data["easting"], data["northing"], data["height"])
    ]
    return data


def egms_csv_to_gdf(file_path: os.PathLike) -> Tuple[gp.GeoDataFrame, str]:
    """Imports the csv in EGMS format and converts it to a geodatabase in the correct CRS.

    :param file_path: The path to the csv file
    :type file_path: os.PathLike
    :return: The data as a geodataframe and the associated layer (EW or vertical)
    :rtype: Tuple[gp.GeoDataFrame, str]
    """
    """Imports the csv in EGMS format and converts it to a geodatabase in the correct CRS.

    :param file_path: The path to the csv file
    :type file_path: os.PathLike
    :return: The data as a geodataframe
    :rtype: gp.GeoDataFrame
    """
    fname = os.path.split(file_path)[-1]
    if "_E_" in fname or fname.endswith("_E.csv"):
        layer = "East-West"
    elif "_U_" in fname or fname.endswith("_U.csv"):
        layer = "Vertical"
    else:
        ValueError(
            "Filename does not contain direction of movement (_E_ or _U_)"
        )
    logging.info("Starting import")
    data = load_csv(file_path)
    logging.info("Creating GeoDataFrame")
    gdf = gp.GeoDataFrame(data=data, crs="EPSG:3035").to_crs("EPSG:32632")
    return gdf, layer


def egms_csv_list_to_gdf(
    file_list: list[os.PathLike],
) -> Tuple[gp.GeoDataFrame, gp.GeoDataFrame]:
    """Loads all files from the file list and returns two geodataframes, one vertical and one for east-west component.

    :param file_list: The list of csv files to read.
    :type file_list: list[os.PathLike]
    :return: _description_
    :rtype: Tuple[gp.GeoDataFrame,gp.GeoDataFrame]
    """
    logging.info("Converting list to geodataframes")
    all_gdf = {"East-West": [], "Vertical": []}
    for file_path in tqdm(
        file_list, desc="Converting EGMS Files", leave=False
    ):
        gdf, layer = egms_csv_to_gdf(file_path)
        all_gdf[layer].append(gdf)
    merged_v = merge_gdf_list(all_gdf["Vertical"])
    merged_ew = merge_gdf_list(all_gdf["East-West"])
    return (merged_v, merged_ew)


def merge_gdf_list(gdf_list: list[gp.GeoDataFrame]) -> gp.GeoDataFrame:
    """Merges all geodataframes in a list to a single geodataframe. Uses "inner join" which only keeps the keys that are present in all geodataframes.

    :param gdf_list: A list of geodataframe objects for merging.
    :type gdf_list: list
    :return: _description_
    :rtype: gp.GeoDataFrame
    """
    logging.info("Merging GeoDataFrames")
    merged = gdf_list[0]
    for ii in range(1, len(gdf_list)):
        merged = merged.merge(gdf_list[ii], how="outer")

    return merged


def buf_count_newlines_gen(fname: os.PathLike) -> int:
    """Counts the number of lines in a text file.
    Adapted from https://stackoverflow.com/questions/845058/how-to-get-the-line-count-of-a-large-file-cheaply-in-python.

    :param fname: The input file
    :type fname: str
    :return: The number of lines in the input file.
    :rtype: int
    """
    logging.info(f"Counting lines in {fname}")

    def _make_gen(reader):
        b = reader(2**16)
        while b:
            yield b
            b = reader(2**16)

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count
