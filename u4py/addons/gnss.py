"""
Contains functions for handling GNSS datasets.
"""
import os
from typing import Callable, Tuple

import geopandas as gp
import shapely as shp

import u4py.utils.files as u4files


def sitelog_to_shp(folder_path: os.PathLike, shp_path: os.PathLike):
    """Converts all sitelogs in the given folderpath from the International GNSS Service to a shapefile.

    :param folder_path: The path to a folder containing sitelogs
    :type folder_path: os.PathLike
    :param shp_path: The path to the shapefile
    :type shp_path: os.PathLike
    """

    log_filelist = u4files.get_file_list(
        filetype=".log", folder_path=folder_path
    )
    data = {"geometry": [], "name": [], "ID": []}
    for fp in log_filelist:
        name, x, y, z = read_sitelog(fp)
        data["ID"].append(os.path.splitext(os.path.split(fp)[-1])[0].upper())
        data["name"].append(name)
        data["geometry"].append(shp.Point(x, y, z))

    gdf = gp.GeoDataFrame(data, crs="EPSG:5332").to_crs("EPSG:32632")
    gdf.to_file(shp_path)


def get_station_coordinates(
    folder_path: os.PathLike, overwrite: bool = False
) -> gp.GeoDataFrame:
    """Loads station coordinates from the given folder. If no shape file with the coordinates is found, it is created.

    :param folder_path: The path to a folder containing sitelogs
    :type folder_path: os.PathLike
    :param overwrite: Whether to overwrite existing data.
    :type overwrite: bool
    :return: A geodataframe for plotting
    :rtype: gp.GeoDataFrame
    """
    shp_path = os.path.join(folder_path, "gnss_stations.shp")
    if not os.path.exists(shp_path) or overwrite:
        sitelog_to_shp(folder_path, shp_path)

    stations = gp.read_file(shp_path)

    return stations


def read_sitelog(file_path: os.PathLike) -> Tuple[str, float, float, float]:
    """Reads a sitelog and returns name and coordinates of the station.

    :param file_path: The path to the sitelog file
    :type file_path: os.PathLike
    :return: A tuple containing (name, x, y, z)
    :rtype: Tuple[str, float, float, float]
    """

    with open(file_path, "rt", newline="\n") as sitelog:
        for row in sitelog:
            row = row.replace("\n", "")
            row = row.replace("\r", "")
            if "Site Name" in row:
                name = get_data(row, str).title()
            elif "X coordinate (m)" in row:
                x = get_data(row, float)
            elif "Y coordinate (m)" in row:
                y = get_data(row, float)
            elif "Z coordinate (m)" in row:
                z = get_data(row, float)
    return name, x, y, z


def get_data(in_str: str, out_dtype: Callable) -> str | float:
    """Gets the data from the row

    :param in_str: The string of the row containing the data
    :type in_str: str
    :param out_dtype: The function to convert the data.
    :type out_dtype: Callable
    :return: A string or float, depending on the
    :rtype: str | float
    """

    return out_dtype(in_str.split(": ")[-1])
