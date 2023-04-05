"""
Contains functions to work with climate data.
"""

import os
from datetime import datetime

import numpy as np


def load_climate_data(file_path: os.PathLike) -> dict:
    """Loads the weather data from a climate archive dataset.

    Arguments:
        file_path -- The text file containing the data

    Returns:
        A dictionary including temperature (avg, min, max) and rainfall.
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
