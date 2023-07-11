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

import logging
import os
import pickle
from multiprocessing import Pool
from typing import Callable, Tuple

import numpy as np
from scipy import optimize as spopt
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def get_processing_results(
    file_list: list[os.PathLike], overwrite: bool = False
) -> Tuple[dict, int]:
    """Checks if processing results are there and reprocesses if not or
    overwrite=True

    :param file_list: A list of file paths to process.
    :type file_list: list[os.PathLike]
    :param overwrite: Overwrite existing results, defaults to False
    :type overwrite: bool, optional
    :return: A tuple containing the processing results and the chunk size.
    :rtype: Tuple[dict, int]
    """
    folder_path = os.path.split(file_list[0])[0]
    base_path = os.path.split(os.path.split(folder_path)[0])[0]
    result_folder = os.path.join(base_path, "INSAR_results")
    if "ASCE" in folder_path:
        pickle_path = "ASCE_processing_results.pkl"
    elif "DESC" in folder_path:
        pickle_path = "DESC_processing_results.pkl"
    elif "BBD_Vert" in folder_path:
        pickle_path = "BBD_Vert_processing_results.pkl"
    elif "BBD_EW" in folder_path:
        pickle_path = "BBD_EW_processing_results.pkl"
    elif "merged" in folder_path:
        pickle_path = "inversion_results.pkl"
    result_path = os.path.join(result_folder, pickle_path)
    if os.path.exists(result_path) and not overwrite:
        results, chunk_size = u4files.load_pickled_results(result_path)
    else:
        results, chunk_size = process_file_list(file_list)
        os.makedirs(result_folder, exist_ok=True)
        with open(result_path, "wb") as pkl_file:
            pickle.dump((results, chunk_size), pkl_file)
    return results, chunk_size


def simple_file_process(
    file_path: os.PathLike, overwrite: bool = False
) -> Tuple[float, float, Tuple]:
    """Does a simple processing for a single file using the simple linear
    decomposition.

    :param file_path: The path to the file to be processed.
    :type file_path: os.PathLike
    :param overwrite: Overwrite existing results, defaults to False
    :type overwrite: bool, optional
    :return: A Tuple containing the x, y coordinates and a Tuple of results (1 linear and 3 sinusoidal components)
    :rtype: Tuple[float, float, Tuple]
    """
    base_path, fname = os.path.split(file_path)
    data = u4files.load_hdf5(file_path)
    if "xmid" not in data.keys():
        fname, ext = os.path.splitext(fname)
        split_name = fname.split("_")
        xmid = int(split_name[2][1:]) + 500
        ymid = int(split_name[3][1:]) + 500
        data["xmid"] = xmid
        data["ymid"] = ymid
    else:
        xmid = data["xmid"]
        ymid = data["ymid"]

    if "time_components" not in data.keys() or overwrite:
        try:
            time = data["time"]
        except KeyError:
            print(file_path)
        time_days = np.linspace(0, len(time) * 6, len(time))
        y = np.nanmedian(data["timeseries"], axis=0)
        slc = np.nonzero(np.isfinite(y))
        time_components = simple_decomposition(time_days[slc], y[slc])
        data["time_components"] = time_components
        u4convert.dict_to_hdf5(file_path, data)
    else:
        time_components = data["time_components"]

    return (xmid, ymid, time_components)


def process_file_list(
    file_list: list, fnc: Callable = simple_file_process
) -> Tuple[list, int]:
    """Uses parallel processing to process a list of files.

    :param file_list: A list of file paths to process.
    :type file_list: list[os.PathLike]
    :param overwrite: Overwrite existing results, defaults to False
    :type overwrite: bool, optional
    :return: A tuple containing a list of processing results and chunk size.
    :rtype: Tuple[list, int]

    Supported functions for processing are:

        - :func:`simple_file_process` for a linear+sinus fit
        - :func:`invert_file` for a full inversion with GrAtSiD

    Multiprocessing in this function uses an unordered mapping through
    `imap_unordered`, so when new functions are implemented, processing must
    be done independently from a single file. Results from "neighboring"
    cells, or files, cannot be considered at runtime.

    """
    with Pool(u4config.cpu_count) as p:
        results = list(
            tqdm(
                p.imap_unordered(fnc, file_list),
                total=len(file_list),
                desc="Processing files",
                leave=False,
            )
        )
    data = u4files.load_hdf5(file_list[0])
    try:
        chunk_size = data["chunk_size"]
    except KeyError:
        chunk_size = 0
    return (results, chunk_size)


def simple_decomposition(time_days: np.ndarray, y: np.ndarray) -> Tuple[float]:
    """Separates signal into linear and sinusoidal part and returns the components.

    :param time_days: Time axis in days.
    :type time_days: np.ndarray
    :param y: The data to be fit.
    :type y: np.ndarray
    :return: The parameters as a tuple of floats (linear slope, amplitude, width, shift).
    :rtype: Tuple[float]
    """
    lin_popt, _, _ = linear_component(time_days, y)
    sin_popt, _, _ = sinus_component(
        time_days, y - u4other.poly1(time_days, *lin_popt)
    )
    return (lin_popt[0], sin_popt[0], sin_popt[1], sin_popt[2])


def get_decomposed_signals(
    data: dict, save_path: os.PathLike = "", overwrite: bool = False
) -> dict:
    """Decomposes signal similar to :func:`simple_decomposition` but also
    returns the fit data.

    :param data: A dictionary containing the time series data of several stations (from chunked data).
    :type data: dict
    :param save_path: The path where to save the results in a pickle, defaults to ""
    :type save_path: os.PathLike
    :param overwrite: Overwrite existing results if True, defaults to False
    :type overwrite: bool, optional
    :return: A dictionary containing the results.
    :rtype: dict

    The results are stored in the dictionary as follows:

        | *"time"*: The time axis,
        | *"y_med"*: The median of displacement of all stations in the dataset,
        | *"fits"*:
        |   *"lin"*: Components of the linear fit,
        |   *"sin"*: Components of the sinusoidal fit
        | *"signals"*:
        |   *"lin"*: The linear model,
        |   *"sin"*: The sinusiodal model,
        |   *"res"*: The residuals of data-(lin+sin),
    """
    # Data Preparation
    time = data["time"]
    y_med = np.nanmedian(data["timeseries"], axis=0)

    # Remove non finite elements
    slc = np.nonzero(np.isfinite(y_med))
    time = time[slc]
    y_med = y_med[slc]

    # Convert time to days
    time_days = np.linspace(0, (time[-1] - time[0]).days, len(time))

    # Linear component
    lin_popt, _, _ = linear_component(time_days, y_med)
    linear = u4other.poly1(time_days, *lin_popt)
    y_detrend = y_med - linear

    # Seasonal component
    sin_popt, _, _ = sinus_component(time_days, y_detrend)
    sinus = u4other.cosinefunc(time_days, *sin_popt)
    y_residual = y_detrend - sinus

    results = {
        "time": time,
        "y_med": y_med,
        "fits": {"lin": lin_popt, "sin": sin_popt},
        "signals": {
            "lin": linear,
            "sin": sinus,
            "res": y_residual,
        },
    }
    return results


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


def get_common_files(
    base_path: os.PathLike, path1: str = "ASCE", path2: str = "DESC"
) -> list:
    """Goes through the ASCE and DESC folders in base_path to find matches.

    :param base_path: The path to the folder where both folder are located.
    :type base_path: os.PathLike
    :param path1: The folder name of the ascending orbit, defaults to "ASCE"
    :type path1: str, optional
    :param path2: The folder name of the descending orbit, defaults to "DESC"
    :type path2: str, optional
    :return: A list of h5 files that are found in both folders.
    :rtype: list
    """
    first_path = os.path.join(base_path, path1)
    second_path = os.path.join(base_path, path2)
    file_list_1 = [f for f in os.listdir(first_path) if f.endswith(".h5")]
    file_list_2 = [f for f in os.listdir(second_path) if f.endswith(".h5")]

    file_list_set = set(file_list_1)
    common_files = [f for f in file_list_2 if f in file_list_set]
    return common_files


def get_inversion_results(
    file_list: list, result_path: os.PathLike = "", overwrite: bool = False
) -> Tuple[dict, int]:
    """Gets the inversion results for the files in file list.

    :param file_list: A list of files to get the inversion results from.
    :type file_list: list
    :param file_list: The path of the results file, defaults to ""
    :type file_list: os.PathLike
    :param overwrite: Overwrite existing results if True, defaults to False
    :type overwrite: bool, optional
    :return: A tuple containing a dictionary with the inversion results and chunk size.
    :rtype: Tuple[dict, int]

    The function first checks if a file named `inversion_results.pkl` is
    located in the folder one level above the files in the list. If that is
    not the case or if `overwrite=True` it processes all files in the list and
    creates this file.
    """
    if not result_path:
        folder_path = os.path.split(file_list[0])[0]
        base_path = os.path.split(os.path.split(folder_path)[0])[0]
        result_folder = os.path.join(base_path, "INSAR_results")
        result_path = os.path.join(result_folder, "inversion_results.pkl")
    else:
        result_folder = os.path.split(result_path)[0]
    if os.path.exists(result_path) and not overwrite:
        with open(result_path, "rb") as pkl_file:
            results, chunk_size = pickle.load(pkl_file)
    else:
        results, chunk_size = process_file_list(file_list, fnc=invert_file)
        result_path = os.path.join(
            result_folder, f"inversion_results_{chunk_size}.pkl"
        )
        os.makedirs(result_folder, exist_ok=True)
        with open(result_path, "wb") as pkl_file:
            pickle.dump((results, chunk_size), pkl_file)
    return results, chunk_size


def invert_file(
    file_path: os.PathLike, overwrite: bool = False
) -> Tuple[float, float, np.ndarray]:
    """Uses Inversion to invert the timeseries into several components.

    :param file_path: The path to the file to invert.
    :type file_path: os.PathLike
    :param overwrite: Overwrite existing data if True, defaults to False
    :type overwrite: bool, optional
    :return: The parameters as a tuple of x,y coordinates and the resulting model parameters in a matrix.
    :rtype: Tuple[float, float, np.ndarray]

    The full processing follows a multi-step procedure:

        1. Load data from `file_path` and format for inversion
        2. Check if inversion data is present. Load that and return, otherwise or if `overwrite` is True continue.
        3. Do a first fit, adding the results to the data matrix.
        4. Find datapoints that are more than 2 standard deviations away.
        5. Do a second fit, filtering out values found in step 4.
        6. Add the results to the data and store it in the original file
        7. Return the point and components.

    A documentation of the algorithm is given in :func:`u4py.analysis.inversion.invert_time_series`.
    """
    _, fname = os.path.split(file_path)
    data = u4files.get_data_for_inversion(file_path)
    matrix = None

    if "inversion_results" not in data.keys() or overwrite:
        _, data, _ = u4invert.invert_time_series(data)
        ind = u4invert.remove_outliers(data["ori_dhat_data"], threshold=2)
        matrix, data, _ = u4invert.invert_time_series(data, ind=ind)
        orig_data = u4files.load_hdf5(file_path)
        orig_data["inversion_results"] = matrix
        u4convert.dict_to_hdf5(file_path, orig_data)
    else:
        matrix = data["inversion_results"]

    return (data["xmid"], data["ymid"], matrix)


def invert_psi_dict(
    data: dict, save_path: os.PathLike = "", overwrite: bool = False
) -> list:
    """Inverts a dictionary loaded or merged from h5-files

    :param data: Data dictionary according to u4py standard (e.g. read from h5).
    :type data: dict
    :param save_path: The path where to save the results in a pickle.
    :type save_path: os.PathLike
    :param overwrite: Overwrite existing results if True, defaults to False
    :type overwrite: bool, optional
    :return: The matrix of components.
    :rtype: list
    """
    if overwrite or not os.path.exists(save_path):
        prepared_data = u4invert.reformat_dict(data)
        ori_matrix, prepared_data, t_1 = u4invert.invert_time_series(
            prepared_data
        )
        ind = u4invert.remove_outliers(
            prepared_data["ori_dhat_data"], threshold=2
        )
        matrix, prepared_data, t_2 = u4invert.invert_time_series(
            prepared_data, ind=ind
        )
        prepared_data["ori_inversion_results"] = ori_matrix
        prepared_data["inversion_results"] = matrix
        prepared_data["ori_dhat_data"]["t"] = t_1
        prepared_data["dhat_data"]["t"] = t_2

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
