"""
Functions for working with river level data
"""

import locale
import os
from datetime import datetime
from typing import Tuple


def _date2num_rhine(in_date: str) -> datetime:
    """Converts a date in river level data to datetime

    :param in_date: input date as string
    :type in_date: str
    :return: time as datetime object
    :rtype: datetime
    """
    locale.setlocale(locale.LC_ALL, "de_DE")
    date = datetime.strptime(in_date, "%b %Y")
    return date


def load_rhine_date(file_path: os.PathLike) -> Tuple[list, list]:
    """Loads rhine water level data downloaded from Düsseldorf city

    :param file_path: The input file.
    :type file_path: os.PathLike
    :return: A Tuple containing to lists: (`date`, `level`)
    :rtype: Tuple[list, list]
    """
    with open(file_path, "rt") as rhinefile:
        _ = rhinefile.readline()
        date = []
        level = []
        for row in rhinefile.readlines():
            row_text = row.split(",")
            date.append(_date2num_rhine(row_text[0]))
            level.append(float(row_text[3].replace("\n", "").replace(" ", "")))
    return (date, level)
