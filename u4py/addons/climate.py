"""
Contains functions to work with climate data.
"""

import os
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
