"""
Contains functions for various file or data conversion operations.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Iterable

import geopandas as gp
import numpy as np

from u4py.utils.types import InversionData


def gnss_dat_to_dict(file_in: os.PathLike | str) -> dict:
    """Converts a gnss dat file to dictionary

    :param file_in: The file to convert.
    :type file_in: os.PathLike | str
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


def get_floatyear(
    time: str | datetime | list | np.ndarray,
) -> float | list | np.ndarray:
    """Converts from an ISO date or datetime to a float based year.

    :param time: The input timestamp to convert.
    :type time: str | datetime | list |np.ndarray
    :raises NotImplementedError: Raised when the input time format is not supported.
    :return: The timestamp as a float based year.
    :rtype: float | list | np.ndarray
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


def get_datetime(
    time: float | list | np.ndarray,
) -> list | np.ndarray | datetime:
    """Converts from a float based year to datetime.

    :param time: The input timestamp to convert.
    :type time: float | list |np.ndarray
    :raises NotImplementedError: Raised when the input time format is not supported.
    :return: The timestamp as a datetime.
    :rtype: float | list |np.ndarray
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
    results: InversionData,
    directions: list = [
        "U",
    ],
) -> dict:
    """Takes the output of :func:`u4py.analyis.processing.invert_psi_dict` and converts it to a more intuitive format.

    :param results: The input dictionary
    :type results: InversionData
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


def reformat_gdf_to_dict(gdf: gp.GeoDataFrame) -> dict:
    """Reformats the geodataframe into a default dictionary format for data
    processing. The output matches the legacy format of sql imported data.


    :param gdf: The dataframe to convert
    :type gdf: gp.GeoDataFrame
    :return: The reformatted dictionary
    :rtype: dict
    """
    filter_keys = [
        "OBJECTID",
        "Shape",
        "ID",
        "Input",
        "X",
        "Y",
        "Z",
        "mean_velo_vert",
        "var_mean_velo_vert",
        "mean_velo_",
        "var_mean_v",
        "mean_velo_east",
        "var_mean_velo_east",
        "mean_velo_",
        "var_mean_v",
        "fid",
        "geom",
        "pid",
        "easting",
        "northing",
        "height",
        "rmse",
        "mean_velocity",
        "mean_velocity_std",
        "acceleration",
        "acceleration_std",
        "seasonality",
        "seasonality_std",
        "geometry",
    ]
    x_str = ""
    y_str = ""
    z_str = ""
    id_key = ""
    mv_key = ""
    mvs_key = ""
    if "X" in gdf.keys():
        x_str = "X"
        y_str = "Y"
        z_str = "Z"
    elif "easting" in gdf.keys():
        x_str = "easting"
        y_str = "northing"
        z_str = "height"

    if "ID" in gdf.keys():
        id_key = "ID"
    elif "pid" in gdf.keys():
        id_key = "pid"
    if "mean_velo_vert" in gdf.keys():
        mv_key = "mean_velo_vert"
        mvs_key = "var_mean_velo_vert"
    elif "mean_velo_" in gdf.keys():
        mv_key = "mean_velo_"
        mvs_key = "var_mean_v"
    if "mean_velo_east" in gdf.keys():
        mv_key = "mean_velo_east"
        mvs_key = "var_mean_velo_east"
    elif "mean_velocity" in gdf.keys():
        mv_key = "mean_velocity"
        mvs_key = "mean_velocity_std"
    if x_str and y_str and z_str and id_key and mv_key and mvs_key:
        output = {
            "x": gdf[x_str].to_numpy(),
            "y": gdf[y_str].to_numpy(),
            "z": gdf[z_str].to_numpy(),
            "ps_id": gdf[id_key].to_numpy(),
            "mean_vel": gdf[mv_key],
            "var_mean_vel": gdf[mvs_key],
            "num_points": len(gdf),
        }
        time = np.array(
            [sql_key_to_time(kk) for kk in gdf.keys() if kk not in filter_keys]
        )
        if len(time) > 0:
            timeseries = np.array(
                [
                    gdf[kk].to_numpy()
                    for kk in gdf.keys()
                    if kk not in filter_keys
                ]
            ).T
            timeseries[timeseries is None] = np.nan

            output.update(
                {"time": time, "timeseries": np.array(timeseries, dtype=float)}
            )
        return output
    else:
        return dict()


def sql_key_to_time(key: str) -> datetime:
    """Returns a datetime object for the given date in sql format.

    :param key: The date as given in the database.
    :type key: str
    :return: A datetime object of the string.
    :rtype: datetime
    """
    try:
        # For BBD 2023 data
        dts = datetime.strptime(key, "date_%Y%m%d")
    except ValueError:
        try:
            # For EGMS data
            dts = datetime.strptime(key, "%Y%m%d")
        except ValueError:
            # For BBD 2021 data
            start_time = datetime(2015, 4, 7)
            dt = timedelta(6)
            ts_0 = 20150
            cur_ts = int(key[5:])
            num_dt = cur_ts - ts_0
            dts = start_time + num_dt * dt

    return dts
