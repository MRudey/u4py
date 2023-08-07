"""
Functions for working with gas storage data by MND Energies
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import numpy as np


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
