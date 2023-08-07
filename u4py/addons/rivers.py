"""
Functions for working with river level data
"""
from __future__ import annotations

import csv
import locale
import os
import pickle as pkl
from datetime import datetime
from typing import Tuple

import numpy as np
from tqdm import tqdm

import u4py.utils.files as u4files


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


def get_pegel_data(file_path: os.PathLike, overwrite: bool = False) -> dict:
    """Loads river level data from a dataset delivered by HLNUG. Loads from a pickeled file if it exists.

    :param file_path: The path to the csv or pkl file with the data.
    :type file_path: os.PathLike
    :param overwrite: Overwrite the pkl file if csv is given, defaults to False
    :type overwrite: bool, optional
    :return: Data in a dictionary.
    :rtype: dict

    *Format of each station:*

        | data = {
        |     "name": *The name of the station*,
        |     "water": *The name of the river/stream*,
        |     "level": {  *The water level data*
        |         "time": *in datetime*,
        |         "level": *in cm*,
        |     },
        |     "temperature": {  *The water temperature, optional*
        |         "time": *in datetime*,
        |         "temperature": *in °C*,
        |     },
        | }
    """

    if file_path.endswith(".pkl") and os.path.exists(file_path):
        with open(file_path, "rb") as pkl_file:
            return pkl.load(pkl_file)
    elif file_path.endswith(".zrx"):
        base_path, _ = os.path.split(file_path)
        pkl_path = os.path.join(base_path, "water_level_data.pkl")
        if os.path.exists(pkl_path) and not overwrite:
            return get_pegel_data(pkl_path)
        else:
            file_list = u4files.get_file_list(
                filetype=".zrx", folder_path=base_path
            )
            return _convert_level_zrx(file_list, pkl_path)


def _convert_level_zrx(
    file_list: list[os.PathLike], output_path: os.PathLike
) -> dict:
    """Converts a list of zrx files to a single pickle file containing all stations.

    :param file_list: The file list
    :type file_list: list[os.PathLike]
    :param output_path: The path where to save the data.
    :type output_path: os.PathLike
    :return: A dictionary of all stations.
    :rtype: dict
    """
    stations = dict()
    for file_path in tqdm(file_list, desc="Loading level data:"):
        data = _load_pegel_from_file(file_path)
        stations[data["station"]] = data
    with open(output_path, "wb") as pkl_file:
        pkl.dump(stations, pkl_file)
    return stations


def _load_pegel_from_file(file_path: os.PathLike) -> dict:
    """Gets the data from a single file. Also loads the temperature data if it exists.

    :param file_path: The path to the water level file.
    :type file_path: os.PathLike
    :return: The water level data, optionally including the water temperature.
    :rtype: dict
    """
    level_data = _load_level_data(file_path)
    temp_data = _load_water_temp_data(file_path)
    merged_data = dict()

    if level_data:
        merged_data["name"] = level_data["name"]
        merged_data["water"] = level_data["water"]
        merged_data["station"] = level_data["station"]
        merged_data["level"] = {
            "time": level_data["time"],
            "level": level_data["level"],
        }
    if temp_data:
        merged_data["temperature"] = {
            "time": temp_data["time"],
            "temperature": temp_data["temperature"],
        }
    return merged_data


def _get_attribute(in_str: str, attrib: str) -> str:
    """Gets the attribute out of a comment string in a zrx file

    :param in_str: The whole string as input.
    :type in_str: str
    :param attrib: The attribute name
    :type attrib: str
    :return: The attribute as a string.
    :rtype: str
    """
    ii_start = in_str.find(attrib) + len(attrib)
    substring = in_str[ii_start:]
    ii_end = substring.find("|*|")
    return substring[:ii_end]


def _load_level_data(file_path: os.PathLike) -> dict:
    """Loads the level data.

    :param file_path: The path to the zrx file
    :type file_path: os.PathLike
    :return: The data as a dictionary
    :rtype: dict
    """
    time = []
    value = []
    with open(file_path, "rt") as csv_file:
        reader = csv.reader(csv_file, delimiter=" ")
        for row in reader:
            if row[0].startswith("#"):
                if "SNAME" in row[0]:
                    name = _get_attribute(row[0], "SNAME")
                if "SWATER" in row[0]:
                    water = _get_attribute(row[0], "SWATER")
                if "SANR" in row[0]:
                    station = _get_attribute(row[0], "SANR")
            else:
                val = float(row[1])
                if val >= 0:
                    time.append(datetime.strptime(row[0], "%Y%m%d%H%M%S"))
                    value.append(val)

    data = {
        "time": np.array(time),
        "level": np.array(value),
        "name": name,
        "water": water,
        "station": station,
    }
    return data


def _load_water_temp_data(file_path: os.PathLike) -> dict:
    """Loads the water temperature data.

    :param file_path: The path to the zrx file
    :type file_path: os.PathLike
    :return: The data as a dictionary
    :rtype: dict
    """
    if "h.Cmd.PWT" not in file_path:
        level_path, file_name = os.path.split(file_path)
        base_path, _ = os.path.split(level_path)
        wtemp_name = file_name.replace("Day.MeanW", "h.Cmd.PWT")
        water_temp_path = os.path.join(
            base_path, "Wassertemperatur", wtemp_name
        )
    else:
        water_temp_path = file_path

    if os.path.exists(water_temp_path):
        time = []
        value = []
        with open(water_temp_path, "rt") as csv_file:
            reader = csv.reader(csv_file, delimiter=" ")
            for row in reader:
                if row[0].startswith("#"):
                    if "SNAME" in row[0]:
                        name = _get_attribute(row[0], "SNAME")
                    if "SWATER" in row[0]:
                        water = _get_attribute(row[0], "SWATER")
                    if "SANR" in row[0]:
                        station = _get_attribute(row[0], "SANR")
                else:
                    val = float(row[1])
                    if val >= 0:
                        time.append(datetime.strptime(row[0], "%Y%m%d%H%M%S"))
                        value.append(val)

        data = {
            "time": np.array(time),
            "temperature": np.array(value),
            "name": name,
            "water": water,
            "station": station,
        }
    else:
        data = False
    return data
