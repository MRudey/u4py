"""
Functions for working with gas storage data by MND Energies
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import geopandas as gp
import numpy as np
import shapely as shp


def _date2num_gas(time_in: str) -> datetime:
    """Converts various time strings formats to ``datetime`` objects.

    :param time_in: The input timestring.
    :type time_in: str
    :return: The datetime object
    :rtype: datetime
    """
    try:
        date = datetime.strptime(time_in, "%d. %m. %Y")
    except ValueError:
        date = datetime.strptime(time_in, "%d.%m.%Y")
    return date


def load_gas_data(
    file_path: os.PathLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the gas storage data from a `csv` file.

    The data can be downloaded from the website of MND Energies in the form of
    a csv file or copied from the webinterface directly.

    :param file_path: The filepath to the csv file.
    :type file_path: os.PathLike
    :return: A tuple containing three numpy arrays:

        | `date`: The date as an array of datetime objects,
        | `level`: Fill level in kWh of gas,
        | `level_perc`: The fill level in percent

    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    with open(file_path, "rt") as gasfile:
        _ = gasfile.readline()
        date = []
        level = []
        for row in gasfile.readlines():
            row_text = row.split("\t")
            date.append(_date2num_gas(row_text[0]))
            level.append(float(row_text[1].replace("\n", "").replace(" ", "")))
    level = np.array(level)
    date = np.array(date)
    level_perc = (level / np.max(level)) * 100
    return date, level, level_perc


def convert_gas(
    file_path: os.PathLike,
    out_path: os.PathLike,
    x: int = 464275,
    y: int = 5516255,
    z: float = 89.0,
):
    """Converts the given file with gas storage data to a gpkg file.

    :param file_path: The path to the file
    :type file_path: os.PathLike
    :param x: x coordinate in epsg:25832, defaults to 464275
    :type x: int, optional
    :param y: y coordinate in epsg:25832, defaults to 5516255
    :type y: int, optional
    :param z: z coordinate in epsg:25832, defaults to 89.
    :type z: float, optional
    """
    date_gas, level, level_perc = load_gas_data(file_path)
    fname = os.path.splitext(os.path.split(file_path)[-1])[0]

    timestamps = ["date_" + t.strftime("%Y%m%d") for t in date_gas]
    gdf = get_gdf(x, y, z, timestamps, level)
    # gdf.to_file(
    #     os.path.join(out_path, fname + ".gpkg"), driver="GPKG", layer="level"
    # )
    gdf = get_gdf(x, y, z, timestamps, level_perc)
    # gdf.to_file(
    #     os.path.join(out_path, fname + ".gpkg"),
    #     driver="GPKG",
    #     layer="level_percent",
    # )


def get_gdf(
    x: int, y: int, z: float, timestamps: list, values: np.ndarray
) -> gp.GeoDataFrame:
    """Converts the inputs to a correctly formated geodataframe

    :param x: Coordinate in epsg:25832
    :type x: int
    :param y: Coordinate in epsg:25832
    :type y: int
    :param z: Coordinate in epsg:25832
    :type z: float
    :param timestamps: The timestamps as list of strings
    :type timestamps: list
    :param values: The values to use at the timestamps
    :type values: np.ndarray
    :return: The geodataframe with the data
    :rtype: gp.GeoDataFrame
    """
    assert len(timestamps) == len(
        values
    ), "Time and Values have different length"
    d = dict()
    for ii, kk in enumerate(timestamps):
        d[kk] = values[ii]
        d["x"] = x
        d["y"] = y
        d["z"] = z
        d["geometry"] = [shp.Point(x, y, z)]

    gdf = gp.GeoDataFrame(d, crs="EPSG:25832")
    return gdf
