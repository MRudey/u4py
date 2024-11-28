"""
**Analysis Functions**

This module contains functions for full processing chains of InSAR data.
Usually, they take single files or a list of files as input and get the
analysis results. Currently, a simple linear+sinusoidal fit and a more
complicated inversion scheme (see :func:`u4py.analysis.inversion`) are
implemented. The results are either stored in an external pickle file or
appended to the original file. Most functions first check if results are
already present to avoid a reprocessing every function call. This can be
overwritten by setting the keyword argument `overwrite=True`.
"""

from __future__ import annotations

import logging
import os
import pickle
from multiprocessing import Pool
from typing import Callable, Iterable, Tuple

import geopandas as gp
import shapely as shp
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.io.gpkg as u4gpkg
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert


def invert_psi_dict(
    data: dict = dict(),
    t_AT: list = [],
    t_EQ: list = [],
    t_EX: list = [],
    num_coeffs: int = 1,
    save_path: os.PathLike = "",
    overwrite: bool = False,
    data_mapping: dict = {
        "dataE": "timeseries",
        "dataN": "timeseries",
        "dataU": "timeseries",
    },
) -> list:
    """Inverts a dictionary loaded or merged from h5-files

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param t_AT: A list with times of known antenna offsets, defaults to []
    :type t_AT: list, optional
    :param t_EQ: A list with times of known earthquakes, defaults to []
    :type t_EQ: list, optional
    :param num_coeffs: The number of parameters to use for inversion, defaults to 1
    :type num_coeffs: int, optional
    :param save_path: The path where to save the results in a pickle, defaults to "".
    :type save_path: os.PathLike, optional
    :param overwrite: Overwrite existing results if True, defaults to False
    :type overwrite: bool, optional
    :param data_mapping: A dictionary mapping the three components to the respective keys in `dataset`, defaults mapping all components to `"timeseries"`.
    :type data_mapping: dict, optional
    :return: The matrix of components.
    :rtype: list
    """
    if (overwrite or not os.path.exists(save_path)) and data:
        prepared_data = u4invert.reformat_dict(data, data_mapping=data_mapping)
        (
            ori_matrix,
            prepared_data,
            t_1,
            parameters_list,
        ) = u4invert.invert_time_series(
            prepared_data,
            t_AT=t_AT,
            t_EQ=t_EQ,
            t_EX=t_EX,
            num_coeffs=num_coeffs,
        )
        ind = u4invert.remove_outliers(
            prepared_data["ori_dhat_data"], threshold=2
        )
        (
            matrix,
            prepared_data,
            t_2,
            parameters_list,
        ) = u4invert.invert_time_series(
            prepared_data,
            t_AT=t_AT,
            t_EQ=t_EQ,
            t_EX=t_EX,
            num_coeffs=num_coeffs,
            ind=ind,
        )
        prepared_data["ori_inversion_results"] = ori_matrix
        prepared_data["inversion_results"] = matrix
        prepared_data["ori_dhat_data"]["t"] = t_1
        prepared_data["dhat_data"]["t"] = t_2
        prepared_data["parameters_list"] = parameters_list

        try:
            os.remove(save_path)
        except FileNotFoundError:
            logging.info("No inversion results found. Creating new file.")
        with open(save_path, "wb") as pkl_file:
            pickle.dump(prepared_data, pkl_file)

    elif os.path.exists(save_path):
        _, fname = os.path.split(save_path)
        logging.info(f"Loading from save file: {fname}")
        with open(save_path, "rb") as pkl_file:
            prepared_data = pickle.load(pkl_file)
    else:
        FileNotFoundError("No Inversion data found")

    # Add other components to results dictionary (in case EW or NS data was given.)
    directions = ["U"]
    if data_mapping["dataE"] != data_mapping["dataU"]:
        directions.append("E")
    elif data_mapping["dataN"] != data_mapping["dataU"]:
        directions.append("N")

    ref_data = u4convert.reformat_inversion_results(
        prepared_data, directions=directions
    )
    return ref_data


def inversion_map_worker(data: dict) -> Tuple[Tuple, Tuple]:
    """Worker function to map the inversion of a dictionary with parallel processing.

    :param data: The input data
    :type data: dict
    :return: The results of the first and second fit as a tuple including x and y coordinates.
    :rtype: Tuple[Tuple, Tuple]

    Returns the data as a tuple of (`x`, `y`, `results`) for each direction.
    In case the inversion somehow goes wrong, (None, None) is returned.
    """

    try:
        matrix_ori, data, _, parameters_list = u4invert.invert_time_series(
            data
        )
        ind = u4invert.remove_outliers(data["ori_dhat_data"], threshold=2)
        matrix, data, _, parameters_list = u4invert.invert_time_series(
            data, ind=ind
        )
        results = (
            (data["xmid"], data["ymid"], matrix_ori),
            (data["xmid"], data["ymid"], matrix),
        )
        return results
    except:
        logging.info("Inversion failed")
        return (None, None)


def batch_mapping(fnc_args: Iterable, fnc: Callable, desc: str) -> list:
    """Maps the function over a pool of workers.

    :param fnc_args: The list of data to be processed.
    :type fnc_args: Iterable
    :param fnc: The function to be mapped.
    :type fnc: Callable
    :param desc: The description to show in the progressbar.
    :type desc: str
    :return: The results as a list.
    :rtype: list
    """
    with Pool(u4config.cpu_count) as p:
        results = list(
            tqdm(
                p.imap_unordered(fnc, fnc_args),
                total=len(fnc_args),
                desc=desc,
            )
        )
    return results


def get_extracts(data: dict) -> dict:
    """Gets a list of arguments for :func:`u4py.analysis.processing.inversion_map_worker` used for parallel processing.

    :param data: The data dictionary loaded from `inversion_results_all.pkl`.
    :type data: dict
    :return: The arguments needed to invert the data.
    :rtype: dict
    """
    extracts = [
        extraction_worker(data, ii)
        for ii in tqdm(
            range(len(data["vertikal"]["ps_id"])),
            # range(10000),
            desc="Extracting Data",
        )
    ]

    return extracts


def extraction_worker(data: dict, ii: int) -> dict:
    """Worker for parallel data extraction from the dictionary.

    :param data: The data dictionary.
    :type data: dict
    :param ii: The index of the vertikal dataset to be extracted.
    :type ii: int
    :param jj: The index of the east-west dataset to be extracted.
    :type jj: int
    :return: A dictionary with the reformatted data.
    :rtype: dict
    """
    if data["vertikal"]["ps_id"][ii] != data["Ost_West"]["ps_id"][ii]:
        raise IndexError("Mismatching indices")
    extract = u4convert.reformat_gpkg(
        data["vertikal"]["x"][ii],
        data["vertikal"]["y"][ii],
        data["vertikal"]["z"][ii],
        data["vertikal"]["ps_id"][ii],
        data["vertikal"]["time"],
        data["vertikal"]["timeseries"][ii, :],
        data["Ost_West"]["timeseries"][ii, :],
    )
    return extract


def get_results(
    name: str,
    dataset: str,
    roi: gp.GeoDataFrame,
    processing_path: os.PathLike,
    direction_paths: list[Tuple[os.PathLike, str]],
    overwrite: bool,
    min_psi: int = 0,
):
    """Loads the results of a inversion with two directions.

    :param name: The name of the region of interest (for saving the intermediate results.)
    :type name: str
    :param dataset: The name of the dataset (e.g. "BBD", "EGMS_1", "EGMS_2)
    :type dataset: str
    :param roi: The region of interest.
    :type roi: gp.GeoDataFrame
    :param processing_path: The path where to store the intermediate results
    :type processing_path: os.PathLike
    :param direction_paths: The path to the psi data and the table name as a list of (path, table_name) Tuples
    :type direction_paths: List[Tuple[os.PathLike, str]]
    :param overwrite: Whether to overwrite existing intermediate data.
    :type overwrite: bool
    :param min_psi: Minimum number of psi to use, defaults to 0.
    :type min_psi: int
    """
    if "EGMS" in dataset:
        gpkg_crs = "EPSG:3035"
    else:
        gpkg_crs = ""
    psi_save_path = os.path.join(processing_path, f"{dataset}_{name}.pkl")
    inv_save_path = os.path.join(
        processing_path, f"{dataset}_{name}_invres.pkl"
    )
    roi_gdf = gp.GeoDataFrame(geometry=[roi], crs="EPSG:32632")
    results = []
    if not os.path.exists(psi_save_path) or overwrite:
        logging.info("Getting vertical results")
        data_v = u4gpkg.load_gpkg_data_region(
            roi_gdf,
            direction_paths[0][0],
            direction_paths[0][1],
            gpkg_crs=gpkg_crs,
        )
        logging.info("Getting E-W results")
        data_ew = u4gpkg.load_gpkg_data_region(
            roi_gdf,
            direction_paths[1][0],
            direction_paths[1][1],
            gpkg_crs=gpkg_crs,
        )
        if data_v and data_ew:
            if (data_v["num_points"] > min_psi) and (
                data_ew["num_points"] > min_psi
            ):
                data_v["timeseries_ew"] = data_ew["timeseries"]
                logging.info("Inverting both components")
                results = invert_psi_dict(
                    data_v,
                    save_path=psi_save_path,
                    data_mapping={
                        "dataE": "timeseries_ew",
                        "dataN": "timeseries",
                        "dataU": "timeseries",
                    },
                    overwrite=overwrite,
                )
    else:
        results = invert_psi_dict(
            save_path=psi_save_path,
            data_mapping={
                "dataE": "timeseries_ew",
                "dataN": "timeseries",
                "dataU": "timeseries",
            },
        )
    return results


def extract_profile(
    name: str,
    dataset: str,
    roi: gp.GeoDataFrame,
    processing_path: os.PathLike,
    direction_paths: list[Tuple[os.PathLike, str]],
    overwrite: bool = False,
):
    """Extracts values along a profile in the roi.

    :param name: The name of the region of interest (for saving the intermediate results.)
    :type name: str
    :param dataset: The name of the dataset (e.g. "BBD", "EGMS_1", "EGMS_2)
    :type dataset: str
    :param roi: The region of interest.
    :type roi: gp.GeoDataFrame
    :param processing_path: The path where to store the intermediate results
    :type processing_path: os.PathLike
    :param direction_paths: The path to the psi data and the table name as a list of (path, table_name) Tuples
    :type direction_paths: List[Tuple[os.PathLike, str]]
    """
    if "EGMS" in dataset:
        gpkg_crs = "EPSG:3035"
    else:
        gpkg_crs = "EPSG:32632"
    roi_gdf = gp.GeoDataFrame(geometry=[roi], crs="EPSG:32632")
    results = {
        "vert": None,
        "ew": None,
        "x_v": None,
        "y_v": None,
        "x_ew": None,
        "y_ew": None,
    }
    psi_save_path = os.path.join(processing_path, f"{dataset}_{name}.pkl")
    if not os.path.exists(psi_save_path) or overwrite:
        logging.info("Getting vertical results")
        data_v = u4gpkg.load_gpkg_data_region(
            roi_gdf,
            direction_paths[0][0],
            direction_paths[0][1],
            gpkg_crs=gpkg_crs,
        )
        if data_v:
            points = gp.GeoDataFrame(
                geometry=[
                    shp.Point(x, y) for x, y in zip(data_v["x"], data_v["y"])
                ],
                crs=gpkg_crs,
            ).to_crs("EPSG:32632")
            results["vert"] = data_v["mean_vel"].to_numpy()
            results["x_v"] = points.geometry.x.to_numpy()
            results["y_v"] = points.geometry.y.to_numpy()
        logging.info("Getting E-W results")
        data_ew = u4gpkg.load_gpkg_data_region(
            roi_gdf,
            direction_paths[1][0],
            direction_paths[1][1],
            gpkg_crs=gpkg_crs,
        )
        if data_ew:
            points = gp.GeoDataFrame(
                geometry=[
                    shp.Point(x, y) for x, y in zip(data_ew["x"], data_ew["y"])
                ],
                crs=gpkg_crs,
            ).to_crs("EPSG:32632")
            results["ew"] = data_ew["mean_vel"].to_numpy()
            results["x_ew"] = points.geometry.x.to_numpy()
            results["y_ew"] = points.geometry.y.to_numpy()
        with open(psi_save_path, "wb") as pklfile:
            pickle.dump(results, pklfile)
    else:
        with open(psi_save_path, "rb") as pklfile:
            results = pickle.load(pklfile)
    return results
