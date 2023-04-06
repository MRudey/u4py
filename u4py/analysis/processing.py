import os
import pickle
from multiprocessing import Pool

import numpy as np
from scipy import optimize as spopt
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.utils.config as u4config
import u4py.utils.files as u4files
from u4py.utils.convert import dict_to_hdf5


def get_processing_results(file_list, overwrite=False):
    """
    Checks if processing results are there and reprocesses if not or
    overwrite=True
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
        results, chunk_size = load_pickle(result_path)
    else:
        results, chunk_size = process_file_list(file_list)
        os.makedirs(result_folder, exist_ok=True)
        with open(result_path, "wb") as pkl_file:
            pickle.dump((results, chunk_size), pkl_file)
    return results, chunk_size


def load_pickle(pickle_path: os.PathLike) -> dict:
    """Loads the given pickle file for plotting"""
    with open(pickle_path, "rb") as pkl_file:
        results, chunk_size = pickle.load(pkl_file)
    return results, chunk_size


def simple_file_process(file_path, overwrite=False):
    """
    Does a simple processing for a single file using the simple linear decomposition.
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
        dict_to_hdf5(file_path, data)
    else:
        time_components = data["time_components"]

    return (xmid, ymid, time_components)


def process_file_list(file_list, fnc=simple_file_process):
    """Uses parallel processing to process a list of files"""
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


def simple_decomposition(time_days: np.ndarray, y: np.ndarray) -> tuple:
    """
    Separates signal into linear and sinusoidal part and returns the components
    """
    lin_popt, _, _ = linear_component(time_days, y)
    sin_popt, _, _ = sinus_component(
        time_days, y - u4other.poly1(time_days, *lin_popt)
    )
    return (lin_popt[0], sin_popt[0], sin_popt[1], sin_popt[2])


def get_decomposed_signals(data: dict) -> dict:
    """
    Decomposes signal similar to `simple_decomposition` but also returns the back calculated data.
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
    sinus = u4other.sinefunc(time_days, *sin_popt)
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


def linear_component(x: np.ndarray, y: np.ndarray) -> tuple:
    """Gets the linear fit of x and y"""
    lin_popt, lin_pcov = spopt.curve_fit(u4other.poly1, x, y)
    lin_perr = 2 * np.sqrt(np.diag(lin_pcov))
    return (lin_popt, lin_pcov, lin_perr)


def sinus_component(x: np.ndarray, y: np.ndarray) -> tuple:
    """Gets the sinusoidal fit of x and y, adjusted to psi time in days"""
    try:
        sin_popt, sin_pcov = spopt.curve_fit(
            u4other.sinefunc, x, y, p0=[2, 6 / 365.25, 0]
        )
        sin_perr = 2 * np.sqrt(np.diag(sin_pcov))
    except RuntimeError:
        sin_popt = [np.nan, np.nan, np.nan]
    return (sin_popt, sin_pcov, sin_perr)


def get_common_files(base_path: os.PathLike, path1="ASCE", path2="DESC"):
    """Goes through the ASCE and DESC folders in base_path to find matches

    Args:
        base_path (os.PathLike): Folder path where the ASCE and DESC folder
        are located
    """
    first_path = os.path.join(base_path, path1)
    second_path = os.path.join(base_path, path2)
    file_list_1 = [f for f in os.listdir(first_path) if f.endswith(".h5")]
    file_list_2 = [f for f in os.listdir(second_path) if f.endswith(".h5")]

    file_list_set = set(file_list_1)
    common_files = [f for f in file_list_2 if f in file_list_set]
    return common_files


def get_inversion_results(file_list, overwrite=False):
    """
    Checks if inversion results are there and reprocesses if not or
    overwrite=True
    """
    folder_path = os.path.split(file_list[0])[0]
    base_path = os.path.split(os.path.split(folder_path)[0])[0]
    result_folder = os.path.join(base_path, "INSAR_results")
    result_path = os.path.join(result_folder, "inversion_results.pkl")
    if os.path.exists(result_path) and not overwrite:
        with open(result_path, "rb") as pkl_file:
            results, chunk_size = pickle.load(pkl_file)
    else:
        results, chunk_size = process_file_list(file_list, fnc=invert_file)
        os.makedirs(result_folder, exist_ok=True)
        with open(result_path, "wb") as pkl_file:
            pickle.dump((results, chunk_size), pkl_file)
    return results, chunk_size


def invert_file(file_path, overwrite=False):
    """
    Does full processing for a single file
    """
    base_path, fname = os.path.split(file_path)
    data = u4files.get_data_for_inversion(file_path)
    matrix = None

    if "inversion_results" not in data.keys() or overwrite:
        try:
            _, data, _ = u4invert.invert_time_series(data)
            ind = u4invert.remove_outliers(data["ori_dhat_data"], threshold=2)
            matrix, data, _ = u4invert.invert_time_series(data, ind=ind)
            orig_data = u4files.load_hdf5(file_path)
            orig_data["inversion_results"] = matrix
            dict_to_hdf5(file_path, orig_data)
        except:
            print(f"Inverting {fname} failed")
    else:
        matrix = data["inversion_results"]

    return (data["xmid"], data["ymid"], matrix)
