"""
Contains functions for various file or data conversion operations.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np


def gnss_dat_to_dict(file_in: os.PathLike) -> dict:
    """Converts a gnss dat file to dictionary

    :param file_in: The file to convert.
    :type file_in: os.PathLike
    :return: The data as a dictionary.
    :rtype: dict
    """
    logging.info(f"Converting {file_in} to dictionary")
    raw = np.loadtxt(file_in, unpack=True)

    ind = np.argsort(raw[0])
    time = np.array([gps_week_to_time(r) for r in raw[0]])
    floatyear = np.array([get_floatyear(r) for r in time])

    data = {
        "gps_week": raw[0][ind],
        "gps_datetime": time[ind],
        "gps_floatyear": floatyear,
        "res_north": raw[1][ind],
        "sig_north": raw[2][ind],
        "res_east": raw[3][ind],
        "sig_east": raw[4][ind],
        "res_up": raw[5][ind],
        "sig_up": raw[6][ind],
    }
    return data


def gps_week_to_time(gps_week: float) -> datetime:
    """Converts a gpsweek timestamp to datetime.

    :param gps_week: The gps timestamp referenced to the 01.06.1980
    :type gps_week: float
    :return: The timestamp as a datetime object.
    :rtype: datetime
    """
    days = gps_week * 7
    dt = datetime(1980, 1, 6, 0, 0) + timedelta(days=days)
    return dt


def get_floatyear(time: str | datetime | Iterable) -> float | Iterable:
    """Converts from an ISO date or datetime to a float based year.

    :param time: The input timestamp to convert.
    :type time: str | datetime | Iterable
    :raises NotImplementedError: Raised when the input time format is not supported.
    :return: The timestamp as a float based year.
    :rtype: float | Iterable
    """
    if isinstance(time, str):
        t = datetime_to_floatyear(datetime.fromisoformat(time))
    elif isinstance(time, datetime):
        t = datetime_to_floatyear(time)
    elif isinstance(time, list):
        t = [get_floatyear(n) for n in time]
    elif isinstance(time, np.ndarray):
        t = np.array(get_floatyear(list(time)))
    else:
        raise NotImplementedError(
            f"Converting from {type(time)} is not supported."
        )
    return t


def get_datetime(time: float | Iterable) -> float | Iterable:
    """Converts from a float based year to datetime.

    :param time: The input timestamp to convert.
    :type time: float | Iterable
    :raises NotImplementedError: Raised when the input time format is not supported.
    :return: The timestamp as a datetime.
    :rtype: float | Iterable
    """
    if isinstance(time, (int, float, complex)) and not isinstance(time, bool):
        t = floatyear_to_datetime(time)
    elif isinstance(time, list):
        t = [get_datetime(n) for n in time]
    elif isinstance(time, np.ndarray):
        t = np.array(get_datetime(list(time)))
    else:
        raise NotImplementedError(
            f"Converting from {type(time)} is not supported."
        )
    return t


def datetime_to_floatyear(time: datetime) -> float:
    """Converts a datettime into a floatyear.

    :param time: The timestamp as datetime.
    :type time: datetime
    :return: The floatyear.
    :rtype: float
    """
    return time.year + ((time - datetime(time.year, 1, 1)).days / 365.25)


def floatyear_to_datetime(time: float) -> datetime:
    """Converts a float based year to datetime.

    :param time: The time as a float
    :type time: float
    :return: The datetime of the year.
    :rtype: datetime
    """
    year = int(time)
    days = (time - year) * 365.25
    t = datetime(year, 1, 1) + timedelta(days=days)
    return t


def reformat_inversion_results(
    results: dict,
    directions: list = [
        "U",
    ],
) -> dict:
    """Takes the output of :func:`u4py.analyis.processing.invert_psi_dict` and converts it to a more intuitive format.

    :param results: The input dictionary
    :type results: dict
    :param directions: Which directions to use, defaults to ["U",] which is the only reasonable component for most fits in this project
    :type directions: dict
    :return: The reformatted dictionary.
    :rtype: dict

    The output dictionary contains three arrays with the times for the original data `"t"`, the first fit `"t_fit_1"` and the second fit without outliers `"t_fit_2"`. Additionally, the result for each component is saved with its respective key ["U", "E", "N"]. These nested dictionaries contain the original data `"y"`, the first fit `"y_fit_1"` and the second fit without outliers `"y_fit_2"`, each with their corresponding error denoted by the suffix `"_err"`.
    """
    # Dictionary of key names
    comp_names = ["data", "sigm", "dhat", "dres"]
    inv_res = np.reshape(
        results["inversion_results"],
        (3, int(len(results["inversion_results"]) / 3)),
    )
    ori_inv_res = np.reshape(
        results["ori_inversion_results"],
        (3, int(len(results["ori_inversion_results"]) / 3)),
    )
    dir_ii = {"E": 0, "N": 1, "U": 2}
    output = dict()
    for ii, drc in enumerate(directions):
        cpn = [cn + drc for cn in comp_names]
        output["t"] = get_datetime(results["t"])
        output["t_fit_1"] = get_datetime(results["ori_dhat_data"]["t"][0])
        output["t_fit_2"] = get_datetime(results["dhat_data"]["t"][0])
        output[drc] = {
            "y": results[cpn[0]],
            "y_err": results[cpn[1]],
            "y_fit_1": results["ori_dhat_data"][cpn[2]],
            "y_fit_1_err": results["ori_dhat_data"][cpn[3]],
            "y_fit_2": results["dhat_data"][cpn[2]],
            "y_fit_2_err": results["dhat_data"][cpn[3]],
            "inversion_results": inv_res[ii + dir_ii[drc]],
            "ori_inversion_results": ori_inv_res[ii + dir_ii[drc]],
            "parameters_list": results["parameters_list"],
        }
    return output


def reformat_gpkg(
    x: float,
    y: float,
    z: float,
    ps_id: int,
    time: np.ndarray,
    dataU: np.ndarray,
    dataE: np.ndarray,
) -> dict:
    """Reformats the data found in the loaded `inversion_results_all.pkl` to be used with the inversion algorithm. The data in the file is formatted as loaded from the tables in the gpkg file.

    :param x: The x coordinate.
    :type x: float
    :param y: The y coordinate.
    :type y: float
    :param z: The z coordinate.
    :type z: float
    :param ps_id: The persistent scatterer ID (station number).
    :type ps_id: int
    :param time: The time axis.
    :type time: np.ndarray
    :param dataU: The data for up and down motion.
    :type dataU: np.ndarray
    :param dataE: The data for east/west motion.
    :type dataE: np.ndarray
    :return: A suitably formatted dictionary.
    :rtype: dict
    """
    data = {
        "t": get_floatyear(time),
        "dataE": dataE,
        "dataN": dataE,
        "dataU": dataU,
        "sigmE": np.ones_like(dataE),
        "sigmN": np.ones_like(dataE),
        "sigmU": np.ones_like(dataU),
        "station": ps_id,
        "xmid": x,
        "ymid": y,
        "z": z,
    }
    return data
