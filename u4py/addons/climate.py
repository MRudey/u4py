"""
Contains functions to work with climate data.
"""

from __future__ import annotations

import csv
import os
import pickle as pkl
from datetime import datetime

import geopandas as gp
import numpy as np
import shapely

from u4py.utils.types import WeatherStation


def load_climate_data(file_path: os.PathLike | str) -> WeatherStation:
    """Loads weather data from a climate archive dataset.

    :param file_path: The textfile containing the data.
    :type file_path: os.PathLike | str
    :return: A dictionary containing temperature (avg, min, max) and rainfall:

        | `"time"`: Time as a list of datetime objects,
        | `"mean"`: Mean temperature (°C),
        | `"max"`: Maximum temperature (°C),
        | `"min"`: Minimum temperature (°C),
        | `"rain"`: Rainfall (mm),

    :rtype: WeatherStation
    """
    date, temp_m, temp_x, temp_n, rain = np.loadtxt(
        file_path,
        skiprows=1,
        delimiter=";",
        usecols=(1, 5, 6, 7, 14),
        unpack=True,
    )
    time = [datetime.strptime(str(int(d)), "%Y%m%d") for d in date]
    temp_m[temp_m < -50] = np.nan
    temp_x[temp_x < -50] = np.nan
    temp_n[temp_n < -50] = np.nan
    rain[rain < 0] = np.nan

    climate_data: WeatherStation = {
        "time": time,
        "mean": temp_m,
        "max": temp_x,
        "min": temp_n,
        "rain": rain,
    }

    return climate_data


def _load_rainfall_data(file_path: os.PathLike | str) -> dict:
    """Loads the data and saves it to a pickle file.

    :param file_path: The path to the csv file containing the data.
    :type file_path: os.PathLike | str
    :return: The data as dictionary of time and mm rainfall.
    :rtype: dict
    """
    time = []
    rainfall = []
    with open(file_path, "rt") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        next(reader)  # Skip row 0
        for row in reader:
            val_str = row[3]
            if "-" in val_str:
                val = 0
            else:
                val = float(val_str.replace(",", "."))
            rainfall.append(val)
            time.append(datetime.strptime(row[0], "%d.%m.%Y %H:%M:%S"))

    data = {"time": np.array(time), "rainfall": np.array(rainfall)}
    pkl_path = os.path.splitext(file_path)[0] + ".pkl"
    with open(pkl_path, "wb") as pkl_file:
        pkl.dump(data, pkl_file)

    return data


def _load_temperature_data(file_path: os.PathLike | str) -> dict:
    """Loads the data and saves it to a pickle file.

    :param file_path: The path to the csv file containing the data.
    :type file_path: os.PathLike | str
    :return: The data as dictionary of time and temperatures for each station.
    :rtype: dict
    """

    # Load station coordinates
    station_path = os.path.join(
        os.path.split(file_path)[0], "WeatherStationCoordinates_HLNUG.csv"
    )
    station_coords = load_station_coords_temperature(station_path)

    # Load data CSV
    with open(file_path, "rt") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        next(reader)  # Skip row 0
        next(reader)  # Skip row 1
        next(reader)  # Skip row 2

        # Read stations from header
        header = next(reader)
        stations = [head.split(" / ")[0] for head in header[2:]]

        data = dict()
        data["time"] = []
        for kk in stations:
            data[kk] = []
        # Read data from rows
        for row in reader:
            for ii, val_str in enumerate(row[2:]):
                if val_str == "-":
                    data[stations[ii]].append(np.nan)
                else:
                    data[stations[ii]].append(float(val_str.replace(",", ".")))
            data["time"].append(datetime.strptime(row[0], "%d.%m.%Y"))

    data["coordinates"] = station_coords

    pkl_path = os.path.splitext(file_path)[0] + ".pkl"
    with open(pkl_path, "wb") as pkl_file:
        pkl.dump(data, pkl_file)

    return data


def get_rainfall_data(
    file_path: os.PathLike | str, overwrite: bool = False
) -> dict:  # type: ignore
    """Loads rainfall data from a dataset delivered by HLNUG. Loads from a pickeled file if it exists.

    :param file_path: The path to the csv or pkl file with the data.
    :type file_path: os.PathLike | str
    :param overwrite: Overwrite the pkl file if csv is given, defaults to False
    :type overwrite: bool, optional
    :return: `time` and `rainfall` as arrays.
    :rtype: dict
    """

    if os.path.splitext(file_path)[-1] == ".pkl" and os.path.exists(file_path):
        with open(file_path, "rb") as pkl_file:
            return pkl.load(pkl_file)
    elif os.path.splitext(file_path)[-1] == ".csv":
        pkl_path = os.path.splitext(file_path)[0] + ".pkl"
        if os.path.exists(pkl_path) and not overwrite:
            return get_rainfall_data(pkl_path)
        else:
            return _load_rainfall_data(file_path)


def get_temperature_data(
    file_path: os.PathLike | str, overwrite: bool = False
) -> dict:  # type: ignore
    """Loads temperature data from a dataset delivered by HLNUG. Loads from a pickeled file if it exists.

    :param file_path: The path to the csv or pkl file with the data.
    :type file_path: os.PathLike | str
    :param overwrite: Overwrite the pkl file if csv is given, defaults to False
    :type overwrite: bool, optional
    :return: `time` and `rainfall` as arrays.
    :rtype: dict
    """
    if os.path.splitext(file_path)[-1] == ".pkl" and os.path.exists(file_path):
        with open(file_path, "rb") as pkl_file:
            return pkl.load(pkl_file)
    elif os.path.splitext(file_path)[-1] == ".csv":
        pkl_path = os.path.splitext(file_path)[0] + ".pkl"
        if os.path.exists(pkl_path) and not overwrite:
            return get_temperature_data(pkl_path)
        else:
            return _load_temperature_data(file_path)


def thermal_expansion(length: float, alpha: float, delta_T: float) -> float:
    """Theoretical thermal expansion of a material.

    :math:`\\Delta L \\approx \\alpha L_0 \\Delta T`

    :param length: The length of the structure.
    :type length: float
    :param alpha: The coefficient of thermal expansion
    :type alpha: float
    :param delta_T: The temperature change.
    :type delta_T: float
    :return: Thermal expansion of a material.
    :rtype: float
    """
    return alpha * length * delta_T


def load_station_coords_temperature(file_path: os.PathLike | str) -> dict:
    """Loads a dictionary with station coordinates from a csv file containing coordinates for weather stations.

    Also saves it as a shapefile.

    :param file_path: Path to the file.
    :type file_path: os.PathLike | str
    :return: Station coordinates
    :rtype: dict
    """

    with open(file_path, "rt", encoding="utf8") as csv_file:
        reader = csv.reader(csv_file)
        _ = next(reader)
        data = {
            "station": [],
            "lat": [],
            "lon": [],
            "source": [],
            "remark": [],
            "crs": "EPSG:4326",
        }
        for line in reader:
            data["station"].append(line[0])
            data["lat"].append(float(line[1]))
            data["lon"].append(float(line[2]))
            data["source"].append(line[3])
            data["remark"].append(line[4])

    # Create GeoDataFrame
    gdf = gp.GeoDataFrame(
        data=data,
        geometry=[
            shapely.Point(lon, lat)
            for lon, lat in zip(data["lon"], data["lat"])
        ],
        crs=data["crs"],
    )
    shp_path = os.path.splitext(file_path)[0] + ".shp"
    gdf.to_file(shp_path)
    return data
