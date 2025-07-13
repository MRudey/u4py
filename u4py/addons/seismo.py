"""
Functions to read and work with the seismic catalogue of Hessen.
"""

from __future__ import annotations

import operator as op
import os
import pickle as pkl
from datetime import datetime
from typing import Callable

import geopandas as gp
import numpy as np
import openpyxl as opyxl

import u4py.analysis.spatial as u4spatial
from u4py.utils.types import EQStation


def get_eq_catalogue(
    file_path: os.PathLike | str, overwrite: bool = False
) -> dict[str, EQStation]:
    """Gets the earthquake catalogue from the file. Saves it as a pickle file.

    :param file_path: The path to the file.
    :type file_path: os.PathLike | str
    :param overwrite: Whether to overwrite the existing file, defaults to False
    :type overwrite: bool, optional
    :return: The data as a dictionary.
    :rtype: dict
    """
    if str(file_path).endswith(".pkl") and os.path.exists(file_path):
        with open(file_path, "rb") as pkl_file:
            return pkl.load(pkl_file)
    elif str(file_path).endswith(".xlsx"):
        base_path, _ = os.path.split(file_path)
        pkl_path = str(file_path).replace(".xlsx", ".pkl")
        if os.path.exists(pkl_path) and not overwrite:
            return get_eq_catalogue(pkl_path)
        else:
            return _load_eq_catalogue(file_path)
    else:
        raise NotImplementedError("Unsupported Format of EQ database")


def _load_eq_catalogue(file_path: os.PathLike | str) -> dict[str, EQStation]:
    """Loads the full earthquake catalogue from an Excel-Sheet and returns it as a dictionary of arrays, including a Python compatible datetime array.

    :param file_path: The path to the file
    :type file_path: os.PathLike | str
    :return: The data as dictionary.
    :rtype: dict
    """
    wb = opyxl.load_workbook(file_path, read_only=True)
    ws = wb.active
    data = dict()
    for ii, row in enumerate(ws.rows):  # type: ignore
        if ii < 1:
            keys = [cell.value for cell in row]
            for k in keys:
                data[k] = []
        else:
            for jj, cell in enumerate(row):
                data[keys[jj]].append(_convert_entry(cell.value, keys[jj]))  # type: ignore
    for kk in data.keys():
        data[kk] = np.array(data[kk])
    all_ok = _find_wrong_times(data)
    if all_ok:
        data["DATETIME"] = _add_datetime(
            data["JAHR"], data["MONAT"], data["TAG"], data["ZEIT"]
        )
    with open(str(file_path).replace(".xlsx", ".pkl"), "wb") as pkl_file:
        pkl.dump(data, pkl_file)
    return data


def _convert_entry(entry: str, key: str) -> int | float | str:
    """Conterts the entry of the cell with given key to the appropriate dataformat.

    :param entry: The cell value entry.
    :type entry: str
    :param key: The key of the data.
    :type key: str
    :return: The appropriate data format.
    :rtype: int | float | str
    """
    convert = {
        "ID": int,
        "EVENT": str,
        "BUNDESLAND": str,
        "JAHR": int,
        "MONAT": int,
        "TAG": int,
        "ZEIT": str,
        "BREITE": _comma_float,
        "LAENGE": _comma_float,
        "LOKATION": str,
        "HERDTIEFE": _comma_float,
        "LOKALMAGNITUDE": _comma_float,
        "EPIZENTRALINTENSITAET": _comma_float,
        "REFERENZ_1": str,
        "BEMERKUNG": str,
        "MOMENTMAGNITUDE": _comma_float,
        "MAKROSEISMISCHE_MAGNITUDE": _comma_float,
        "MLMUS": _comma_float,
        "MLIH": _comma_float,
        "SR": str,
    }
    return convert[key](entry)


def _comma_float(value: str | float) -> float:
    """Returns the correct float if a comma is present.

    :param value: The value of the cell (may include a comma)
    :type value: str | float
    :return: The cell as a float.
    :rtype: float
    """
    if isinstance(value, str):
        return float(value.replace(",", "."))
    else:
        return value


def _add_datetime(
    years: np.ndarray, months: np.ndarray, days: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Creates a datetime array for better compatability

    :param years: The years array.
    :type years: np.ndarray
    :param months: The months array.
    :type months: np.ndarray
    :param days: The days array.
    :type days: np.ndarray
    :param times: The times array.
    :type times: np.ndarray
    :return: The datetimes as array.
    :rtype: np.ndarray
    """
    times = _clean_timestrings(times)
    dates = np.array(
        [
            datetime.strptime(
                f"{y:04}-{m:02}-{d:02} {t}", "%Y-%m-%d %H:%M:%S.%f"
            )
            for y, m, d, t in zip(years, months, days, times)
        ]
    )
    return dates


def _clean_timestrings(times: np.ndarray) -> np.ndarray:
    """Cleans some simple problems in the timestring array.

    :param times: The time string array.
    :type times: np.ndarray
    :return: The fixed time string array.
    :rtype: np.ndarray
    """
    for ii, t in enumerate(times):
        while len(t) < 10:
            t = _check_time(t)
            times[ii] = t
    return times


def _check_time(t: str) -> str:
    """Checks and fixes simple problems in the timestring.

    :param t: The timestring
    :type t: str
    :return: The fixed timestring
    :rtype: str
    """
    if t == "None":
        return "00:00:00.0"
    elif t[1] == ":":
        return "0" + t
    elif "." not in t:
        return t + ".0"
    else:
        return ""


def _find_wrong_times(data: dict) -> bool:
    """Finds the Event IDs where something is wrong with the timestamp.

    :param data: The data as dictionary.
    :type data: dict
    :return: True if nothing is wrong.
    :rtype: bool
    """
    all_ok = True
    for ii, t in enumerate(data["ZEIT"]):
        if t == "None":
            pass
        else:
            if len(np.argwhere(np.array([*t]) == ".")) > 1:
                print("Multiple decimal separator:", data["ID"][ii])
                all_ok = False
            if len(np.argwhere(np.array([*t]) == ":")) > 2:
                print("Too many time separator:", data["ID"][ii])
                all_ok = False
            if _check_seconds(t):
                print("Seconds too big?", data["ID"][ii])
                all_ok = False
    return all_ok


def _check_seconds(t: str) -> bool:
    """Checks if the seconds timestamp is correct (<= 60)

    :param t: The timestring
    :type t: str
    :return: True if the timestamp is too large.
    :rtype: bool
    """
    last_colon = np.argwhere(np.array([*t]) == ":")[-1][0]
    return float(t[last_colon + 1 :]) >= 60


def cut_catalogue(
    data: dict, key: str, value, condition: Callable = op.ge
) -> dict:
    """Cuts the catalog by the given `key` by `min_val`, e.g. `key="DATETIME"` and `min_val="2014-04-01T00:00"`

    :param data: The catalogue
    :type data: dict
    :param key: The key of the catalogue to use for slicing.
    :type key: str
    :param value: The value to use for slicing.
    :type value: Any
    :param condition: The condition used for slicing, defaults to op.ge (">=")
    :type condition: str, optional
    :return: The cut catalogue
    :rtype: dict
    """
    if key == "DATETIME":
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
    slc = np.argwhere(condition(data[key], value)).flat
    for kk in data.keys():
        data[kk] = data[kk][slc]
    return data


def get_prepared_eq_catalogue(
    file_path: os.PathLike | str,
    bld_path: os.PathLike | str,
    start_time: str | datetime = "2015-04-01T00:00",
    min_magnitude: float = 2.5,
) -> gp.GeoDataFrame:
    """Returns an already cut and cleaned catalogue ready for the project.

    :param file_path: The path to the full catalogue file.
    :type file_path: os.PathLike | str
    :param bld_path: The path to a shapefile containing the outlines of the states in Germany.
    :type bld_path: os.PathLike | str
    :param start_time: The desired starting time of the catalogue, defaults to "2014-01-01T00:00"
    :type start_time: str | datetime, optional
    :param min_magnitude: The desired minimum magnitude, defaults to 2.5
    :type min_magnitude: float, optional
    :return: The sliced and masked catalogue.
    :rtype: gp.GeoDataFrame
    """
    bld = gp.read_file(bld_path).to_crs("EPSG:23032")
    hessen = bld[bld["GEN"] == "Hessen"]
    data = get_eq_catalogue(file_path)
    data = cut_catalogue(data, key="DATETIME", value=start_time)
    data = cut_catalogue(data, key="LOKALMAGNITUDE", value=min_magnitude)
    eq_gdf = u4spatial.catalog_to_gdf(data, mask=hessen)
    return eq_gdf
