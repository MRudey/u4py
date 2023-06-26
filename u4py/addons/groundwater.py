"""
Contains functions to work with groundwater data
"""
import csv
import os
import pickle as pkl
from datetime import datetime

import geopandas as gp
from shapely.geometry import Point
from tqdm import tqdm


def get_groundwater_data(file_path: os.PathLike) -> dict:
    """Loads data from pickled file.

    :param file_path: Path to the pickle file
    :type file_path: os.PathLike
    :return: The data as a dictionary with station names as keys.
    :rtype: dict

    The ground water data was supplied as a large textfile that was converted
    using ``convert_GW_csv()`` to a pickle file. The data is organized in
    nested dictionaries. The outer dictionary uses the station name as
    headers: ``station["NAME"]``. This returns another dictionary containing
    the data:

        | `"shortID"`: Short ID given by HLNUG,
        | `"gruwahID"`: Grundwasser ID given by HLNUG,
        | `"name"`: The name of the station,
        | `"easting"`: Easting in EPSG:31467,
        | `"northing"`: Northing in EPSG:31467,
        | `"time"`: The time as a list of datetime objects,
        | `"height"`: The height of the water table above NN.

    """
    with open(file_path, "rb") as pkl_file:
        stations = pkl.load(pkl_file)
    return stations


def get_stations(stations: dict) -> gp.GeoDataFrame:
    """Converts the stations dictionary to a GeoDataFrame for plotting.

    :param stations: The dictionary of stations.
    :type stations: dict
    :return: GeoDataFrame with the stations as points, including the `name,
    gruwahID,` and `shortID`.
    :rtype: gp.GeoDataFrame
    """

    station_points = {
        "name": [],
        "geometry": [],
        "gruwahID": [],
        "shortID": [],
    }
    for station in stations.values():
        station_points["geometry"].append(
            Point(station["easting"], station["northing"])
        )
        station_points["name"].append(station["name"])
        station_points["gruwahID"].append(station["gruwahID"])
        station_points["shortID"].append(station["shortID"])

    stations_gdf = gp.GeoDataFrame(station_points, crs="EPSG:31467")
    stations_gdf = stations_gdf.to_crs("EPSG:23032")
    return stations_gdf


def convert_GW_csv(file_path: os.PathLike) -> dict:
    """Converts a groundwater csv file provided by HLNUG to a dictionary.

    :param file_path: Path to the csv file.
    :type file_path: os.PathLike
    :return: Dictionary of data data organized in nested dictionaries. The
    outer dictionary uses the station name as headers: ``station["NAME"]``.
    This returns another dictionary containing the data:

        | `"shortID"`: Short ID given by HLNUG,
        | `"gruwahID"`: Grundwasser ID given by HLNUG,
        | `"name"`: The name of the station,
        | `"easting"`: Easting in EPSG:31467,
        | `"northing"`: Northing in EPSG:31467,
        | `"time"`: The time as a list of datetime objects,
        | `"height"`: The height of the water table above NN.
    :rtype: dict
    """

    with open(file_path, "rt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        next(csv_reader, None)  # skip header

        stations = dict()
        first_run = True
        for row in tqdm(csv_reader, desc="Reading entries from csv..."):
            if first_run:
                station = _new_station(row)
            elif station["gruwahID"] == int(row[1]):
                station["time"].append(datetime.strptime(row[5], "%d.%m.%Y"))
                station["height"].append(_dec2float(row[6]))
            else:
                stations[station["name"]] = station
                station = _new_station(row)
            first_run = False
    return stations


def _new_station(row: list) -> dict:
    """Creates a new station from the entries in the row

    :param row: row read from the csv file
    :type row: list
    :return: dictionary with single entry
    :rtype: dict
    """
    try:
        shortID = int(row[0])
    except ValueError:
        shortID = 0
    station = {
        "shortID": shortID,
        "gruwahID": int(row[1]),
        "name": row[2],
        "easting": _dec2float(row[3]),
        "northing": _dec2float(row[4]),
        "time": [datetime.strptime(row[5], "%d.%m.%Y")],
        "height": [_dec2float(row[6])],
    }
    return station


def _dec2float(string: str) -> float:
    """Returns a float from a string containing a comma as decimal separator.

    :param string: the string to be converted
    :type string: str
    :return: float of the string
    :rtype: float
    """
    return float(string.replace(",", "."))
