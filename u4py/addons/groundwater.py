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
    """Loads data from pickled file

    Arguments:
        file_path -- path to the pickle file

    Returns:
        The data as dictionary
    """
    with open(file_path, "rb") as pkl_file:
        stations = pkl.load(pkl_file)
    return stations


def get_stations(stations: dict) -> gp.GeoDataFrame:
    """Gets coordinates from stations and returns a `GeoDataFrame`

    Arguments:
        stations -- dictionary with stations

    Returns:
        `GeoDataFrame` with stations as points
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
    """Opens a CSV file with groundwater data by hlnug into a dictionary

    Arguments:
        file_path -- path of the csv file

    Returns:
        dictionary with the data
    """
    with open(file_path, "rt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        next(csv_reader, None)  # skip header

        stations = dict()
        first_run = True
        for row in tqdm(csv_reader, desc="Reading entries from csv..."):
            if first_run:
                station = new_station(row)
            elif station["gruwahID"] == int(row[1]):
                station["time"].append(datetime.strptime(row[5], "%d.%m.%Y"))
                station["height"].append(dec2float(row[6]))
            else:
                stations[station["name"]] = station
                station = new_station(row)
            first_run = False
    return stations


def new_station(row: list) -> dict:
    """Creates a new station from the entries in the row

    Arguments:
        row -- row read from the csv file

    Returns:
        dictionary with single entry
    """
    try:
        shortID = int(row[0])
    except ValueError:
        shortID = 0
    station = {
        "shortID": shortID,
        "gruwahID": int(row[1]),
        "name": row[2],
        "easting": dec2float(row[3]),
        "northing": dec2float(row[4]),
        "time": [datetime.strptime(row[5], "%d.%m.%Y")],
        "height": [dec2float(row[6])],
    }
    return station


def dec2float(string: str) -> float:
    """Returns a float from a string containing a comma as decimal separator

    Arguments:
        string -- string to be converted

    Returns:
        float of the string
    """
    return float(string.replace(",", "."))
