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

import numpy as np
from scipy import optimize as spopt
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def linear_component(x: np.ndarray, y: np.ndarray) -> Tuple:
    """Gets the linear fit of x and y using scipy.curve_fit.

    :param x: The time axis
    :type x: np.ndarray
    :param y: The data to fit as 1D numpy array.
    :type y: np.ndarray
    :return: A Tuple containing optimized parameters, covariance and 2 sigma
    :rtype: Tuple
    """
    lin_popt, lin_pcov = spopt.curve_fit(u4other.poly1, x, y)
    lin_perr = 2 * np.sqrt(np.diag(lin_pcov))
    return (lin_popt, lin_pcov, lin_perr)


def sinus_component(x: np.ndarray, y: np.ndarray) -> Tuple:
    """Gets the sinusoidal fit of x and y, adjusted to psi time in days using scipy.curve_fit.

    :param x: The time axis
    :type x: np.ndarray
    :param y: The data to fit as 1D numpy array.
    :type y: np.ndarray
    :return: A Tuple containing optimized parameters, covariance and 2 sigma
    :rtype: Tuple
    """
    try:
        sin_popt, sin_pcov = spopt.curve_fit(
            u4other.cosinefunc, x, y, p0=[2, 6 / 365.25, 0]
        )
        sin_perr = 2 * np.sqrt(np.diag(sin_pcov))
    except RuntimeError:
        sin_popt = [np.nan, np.nan, np.nan]
    return (sin_popt, sin_pcov, sin_perr)


def invert_psi_dict(
    data: dict,
    t_AT: list = [],
    t_EQ: list = [],
    t_EX: list = [],
    num_coeffs: int = 1,
    save_path: os.PathLike = "",
    overwrite: bool = False,
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
    :return: The matrix of components.
    :rtype: list
    """
    if overwrite or not os.path.exists(save_path):
        prepared_data = u4invert.reformat_dict(data)
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

    ref_data = u4convert.reformat_inversion_results(prepared_data)
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
                p.imap_unordered(inversion_map_worker, fnc_args),
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
