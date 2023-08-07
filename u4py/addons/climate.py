"""
Contains functions to work with climate data.
"""

import csv
import os
import pickle as pkl
from datetime import datetime

import numpy as np


def load_climate_data(file_path: os.PathLike) -> dict:
    """Loads weather data from a climate archive dataset.

    :param file_path: The textfile containing the data.
    :type file_path: os.PathLike
    :return: A dictionary containing temperature (avg, min, max) and rainfall:

        | `"time"`: Time as a list of datetime objects,
        | `"mean"`: Mean temperature (°C),
        | `"max"`: Maximum temperature (°C),
        | `"min"`: Minimum temperature (°C),
        | `"rain"`: Rainfall (mm),

    :rtype: dict
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

    climate_data = {
        "time": time,
        "mean": temp_m,
        "max": temp_x,
        "min": temp_n,
        "rain": rain,
    }

    return climate_data


def _load_rainfall_data(file_path: os.PathLike) -> tuple:
    """Loads the data and saves it to a pickle file.

    :param file_path: The path to the csv file containing the data.
    :type file_path: os.PathLike
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
    with open(file_path.replace(".csv", ".pkl"), "wb") as pkl_file:
        pkl.dump(data, pkl_file)

    return data


def get_rainfall_data(
    file_path: os.PathLike, overwrite: bool = False
) -> tuple:
    """Loads rainfall data from a dataset delivered by HLNUG. Loads from a pickeled file if it exists.

    :param file_path: The path to the csv or pkl file with the data.
    :type file_path: os.PathLike
    :param overwrite: Overwrite the pkl file if csv is given, defaults to False
    :type overwrite: bool, optional
    :return: `time` and `rainfall` as arrays.
    :rtype: tuple
    """

    if file_path.endswith(".pkl") and os.path.exists(file_path):
        with open(file_path, "rb") as pkl_file:
            return pkl.load(pkl_file)
    elif file_path.endswith(".csv"):
        pkl_path = file_path.replace(".csv", ".pkl")
        if os.path.exists(pkl_path) and not overwrite:
            return get_rainfall_data(pkl_path)
        else:
            return _load_rainfall_data(file_path)


def thermal_expansion(length: float, alpha: float, delta_T: float):
    """Theoretical thermal expansion of a material.

    :math:`\\Delta L \\approx \\alpha L_0 \\Delta T`

    :param length: The length of the structure.
    :type length: float
    :param alpha: The coefficient of thermal expansion
    :type alpha: float
    :param delta_T: The temperature change.
    :type delta_T: float
    """
    return alpha * length * delta_T
