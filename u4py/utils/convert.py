"""
Contains functions for various file or data conversion operations.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from multiprocessing import Manager, Pool
from typing import Iterable

import geopandas
import h5py
import numpy as np
import shapely.geometry as shpgeo
from dbfread import DBF
from numba import jit, prange
from numba_progress import ProgressBar
from tqdm import tqdm

import u4py.utils.config as u4config


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


def dbf_to_dict(file_in: os.PathLike) -> dict:
    """Generic conversion of dbf file to dictionary keeping the names of the original file.

    :param file_in: The file to convert.
    :type file_in: os.PathLike
    :return: The data as a dictionary.
    :rtype: dict
    """
    logging.info(f"Converting {file_in} to dictionary.")
    geodf = geopandas.read_file(file_in)
    if not geodf.crs:
        geodf = geodf.set_crs("EPSG:4326")
    geodf = geodf.to_crs("EPSG:32632")
    output = dict()
    for k in geodf.keys():
        if k == "geometry":
            output[k] = []
            for geom in geodf[k].to_numpy():
                if isinstance(geom, shpgeo.Point):
                    output[k].append((geom.x, geom.y))
                elif isinstance(geom, shpgeo.Polygon):
                    output[k].append(geom.bounds)
            output[k] = np.array(output[k])
        else:
            output[k] = geodf[k].to_numpy()
    return output


def psi_dbf_to_dict(file_in: os.PathLike) -> dict:
    """Takes a dbf file and returns a dictionary with numpy arrays for x, y, z
    coordinates, PS_ID and timeseries for each entry.

    :param file_in: The file to convert.
    :type file_in: os.PathLike
    :return: The data as a dictionary.
    :rtype: dict
    """
    logging.info(f"Converting {file_in} to dictionary.")
    with DBF(file_in) as dbffile:
        # Preallocation
        num_points = len(dbffile)
        xx = np.zeros(num_points)
        yy = np.zeros(num_points)
        zz = np.zeros(num_points)
        ps_id = np.zeros(num_points)

        # Define type of file:
        all_keys = [k for k in dbffile.field_names]
        has_time = True
        if "stack_ID" in all_keys:
            has_time = False
            mean_vel = np.zeros(num_points)
            var_mean_vel = np.zeros(num_points)
        elif "PS_ID" in all_keys:
            non_time_keys = ["X", "Y", "Z", "PS_ID"]
            id_key = "PS_ID"
        if "Input" in all_keys:
            non_time_keys = [
                "X",
                "Y",
                "Z",
                "ID",
                "Input",
                "mean_velo_",
                "var_mean_v",
            ]
            id_key = "ID"
        # Create time axis
        if has_time:
            key_list = [k for k in all_keys if k not in non_time_keys]
            time = np.array([key_to_time(k) for k in key_list])
            num_fields = len(key_list)
            timeseries = np.zeros((num_points, num_fields))
            for ii, record in enumerate(dbffile):
                xx[ii] = record["X"]
                yy[ii] = record["Y"]
                zz[ii] = record["Z"]
                ps_id[ii] = record[id_key]
                for jj, k in enumerate(key_list):
                    timeseries[ii, jj] = record[k]
        else:
            for ii, record in enumerate(dbffile):
                xx[ii] = record["X"]
                yy[ii] = record["Y"]
                zz[ii] = record["Z"]
                ps_id[ii] = record["PS_ID"]
                mean_vel[ii] = record["mean_veloc"]
                var_mean_vel[ii] = record["var_mean_v"]

    if has_time:
        output = {
            "x": xx,
            "y": yy,
            "z": zz,
            "time": time,
            "ps_id": ps_id,
            "timeseries": timeseries,
        }
    else:
        output = {
            "x": xx,
            "y": yy,
            "z": zz,
            "ps_id": ps_id,
            "mean_vel": mean_vel,
            "var_mean_vel": var_mean_vel,
        }
    return output


def key_to_time(
    key: str,
    starttime: datetime = datetime(year=2015, month=4, day=4),
    startindex: int = 20150,
    acqdiff: timedelta = timedelta(days=6),
) -> datetime:
    """Converts a key timestamp to a datetime

    :param key: Timestamp key
    :type key: str
    :param starttime: Time when first PSI data was taken, defaults to datetime(year=2015, month=4, day=4)
    :type starttime: datetime, optional
    :param startindex: First index when counting is started, defaults to 20150
    :type startindex: int, optional
    :param acqdiff: Timedifference between acquisitions, defaults to timedelta(days=6)
    :type acqdiff: timedelta, optional
    :return: converted and referenced datetime
    :rtype: datetime
    """
    timediff = (int(key[5:]) - startindex) * acqdiff
    return starttime + timediff


def chunk_data(
    data: dict,
    save_folder: os.PathLike,
    chunksize: int = 1000,
    min_values: int = 5,
    compress: bool = True,
) -> list:
    """Chunks data into many smaller files with spanning a square of `chunksize` meters. Discards chunks with less than `min_values`.

    :param data: The data loaded from a large unchunked h5 file.
    :type data: dict
    :param save_folder: The folder where to store the smaller files.
    :type save_folder: os.PathLike
    :param chunksize: The chunksize to use for chunking, defaults to 1000
    :type chunksize: int, optional
    :param min_values: Minimum number of PSI points to be present in the chunk, defaults to 5
    :type min_values: int, optional
    :param compress: Whether to compress the data using gzip, defaults to True
    :type compress: bool, optional
    :return: A list of chunks that have not been converted for some reason.
    :rtype: list
    """
    logging.info(f"Chunking data with multiprocessing.")
    xrange = get_bounds(np.min(data["x"]), np.max(data["x"]), chunksize)
    yrange = get_bounds(np.min(data["y"]), np.max(data["y"]), chunksize)
    chunk_list = get_chunks(xrange, yrange)
    os.makedirs(save_folder, exist_ok=True)
    logging.debug("Creating data manager.")
    manager = Manager()
    d = manager.dict(data)

    logging.debug("Creating arguments for parallel data chunking.")
    args = [
        (chunk, chunksize, min_values, compress, save_folder, d)
        for chunk in chunk_list
    ]
    logging.debug(f"Starting parallel pool for data chunking.")
    with Pool(u4config.cpu_count) as p:
        results = list(
            tqdm(
                p.map(chunking_worker, args),
                total=len(args),
                desc="Chunking data",
                leave=False,
            )
        )
    return results


def chunk_data_numba(
    data: dict,
    save_folder: os.PathLike,
    chunksize: int = 1000,
    min_values: int = 5,
    compress: bool = True,
):
    """Chunks data into many smaller files with spanning a square of `chunksize` meters. Discards chunks with less than `min_values`.

    Does the same as :func:`chunk_data` but uses `numba` for parallel processing instead of :func:`Pool.map`.

    :param data: The data loaded from a large unchunked h5 file.
    :type data: dict
    :param save_folder: The folder where to store the smaller files.
    :type save_folder: os.PathLike
    :param chunksize: The chunksize to use for chunking, defaults to 1000
    :type chunksize: int, optional
    :param min_values: Minimum number of PSI points to be present in the chunk, defaults to 5
    :type min_values: int, optional
    :param compress: Whether to compress the data using gzip, defaults to True
    :type compress: bool, optional
    :return: A list of chunks that have not been converted for some reason.
    :rtype: list
    """
    logging.info(f"Chunking data with numba.")
    xrange = get_bounds(np.min(data["x"]), np.max(data["x"]), chunksize)
    yrange = get_bounds(np.min(data["y"]), np.max(data["y"]), chunksize)
    chunk_list = get_chunks(xrange, yrange)
    os.makedirs(save_folder, exist_ok=True)
    total = len(chunk_list)

    with ProgressBar(desc="Chunking Data", total=total) as progress:
        numba_chunking(
            total,
            chunk_list,
            chunksize,
            min_values,
            compress,
            save_folder,
            data,
            progress,
        )


@jit(parallel=True, nopython=False)
def numba_chunking(
    total: int,
    chunk_list: list,
    chunksize: int,
    min_values: int,
    compress: bool,
    save_folder: os.PathLike,
    data: dict,
    progress_proxy: ProgressBar,
):
    """Wrapper for the chunking with numba (needed for progress bar)

    :param total: The total number of iterations
    :type total: int
    :param chunk_list: A coordinate bounds for chunking.
    :type chunk_list: list
    :param chunksize: The chunksize to use for chunking.
    :type chunksize: int
    :param min_values: Minimum number of PSI points to be present in the chunk.
    :type min_values: int
    :param compress: Whether to compress the data using gzip.
    :type compress: bool
    :param save_folder: The folder where to store the smaller files.
    :type save_folder: os.PathLike
    :param data: The data loaded from a large unchunked h5 file.
    :type data: dict
    :param progress_proxy: `ProgressBar` object for updating.
    :type progress_proxy: ProgressBar
    """
    for ii in prange(total):
        args = (
            chunk_list[ii],
            chunksize,
            min_values,
            compress,
            save_folder,
            data,
        )
        chunking_worker(args)
        progress_proxy.update(1)


def chunking_worker(args) -> str:
    """Encapsulated worker for getting a chunk out of the data.

    :param args: The chunking arguments as specified in :func:`numba_chunking`
    :type args: list
    :return: Chunks that have not been converted for some reason.
    :rtype: str
    """
    chunk = args[0]
    chunksize = args[1]
    min_values = args[2]
    compress = args[3]
    save_folder = args[4]
    data = args[5]

    xlow = data["x"] >= chunk[0]
    xhigh = data["x"] < chunk[1]
    ylow = data["y"] >= chunk[2]
    yhigh = data["y"] < chunk[3]
    slc = np.nonzero(xlow * xhigh * ylow * yhigh)

    if len(slc[0]) > min_values:
        chunk_name = "PSI_chunk_x%i_y%i.h5" % (chunk[0], chunk[2])

        logging.debug(f"Reading chunk data for {chunk_name}")
        start_time = time.time()
        output = slice_array(data, slc)
        run_time = time.time() - start_time
        logging.debug(f"Reading {chunk_name} took {run_time:.3} seconds.")

        output["xmid"] = chunk[0] + 0.5 * chunksize
        output["ymid"] = chunk[2] + 0.5 * chunksize
        output["num_points"] = len(slc[0])
        output["chunk_size"] = chunksize
        if "time" in data.keys():
            if compress:
                dict_to_hdf5(os.path.join(save_folder, chunk_name), output)
            else:
                dict_to_hdf5(
                    os.path.join(save_folder, chunk_name),
                    output,
                    compression=None,
                    compression_opts=0,
                )
        else:
            return chunk_name


def slice_array(data: dict, slc_chunk: slice) -> dict:
    """Creates a new dictionary with all data sliced by `slc_chunk`. Used to create the smaller chunked data.

    :param data: The data loaded from a large unchunked h5 file.
    :type data: dict
    :param slc_chunk: A slicing object to extract specific stations from the larger dictionary.
    :type slc_chunk: slice
    :return: A dictionary only containing a subset of the data in a specific chunk.
    :rtype: dict
    """
    output = dict()
    for k in data.keys():
        if k == "time":
            output[k] = data[k]
        else:
            output[k] = data[k][slc_chunk]
    return output


def get_bounds(minval: float, maxval: float, chunksize: int) -> np.ndarray:
    """Takes the minimum and maximum coordinate of a direction and creates an array with the boundaries that are going to be used for the slicing.

    :param minval: The minimum coordinate.
    :type minval: float
    :param maxval: The maximum coordinate.
    :type maxval: float
    :param chunksize: The chunk size
    :type chunksize: int
    :return: An array containing the boundaries.
    :rtype: np.ndarray
    """ """

    """
    minbound = int(np.floor(minval / chunksize) * chunksize)
    maxbound = int(np.ceil(maxval / chunksize) * chunksize)
    out = np.arange(minbound, maxbound, chunksize)
    return out


@jit(nopython=True)
def get_chunks(xrange: np.ndarray, yrange: np.ndarray) -> list:
    """Returns the corners for chunking from a range of x and y values.

    :param xrange: The range of x coordinates.
    :type xrange: np.ndarray
    :param yrange: The range of y coordinates.
    :type yrange: np.ndarray
    :return: A list of corners for chunking.
    :rtype: list
    """
    chunk_list = []
    len_x = len(xrange)
    len_y = len(yrange)

    for ii in range(len_x):
        xm = xrange[ii]
        if ii < len(xrange) - 1:
            for jj in range(len_y):
                ym = yrange[jj]
                if jj < len(yrange) - 1:
                    chunk_list.append([xm, xrange[ii + 1], ym, yrange[jj + 1]])
    return chunk_list


def convert_shapefile(file_path: os.PathLike):
    """Converts the given dbf file into a h5 file. The h5 file only contains the necessary information and is zipped with gzip.

    :param file_path: A shapefile containing PSI data.
    :type file_path: os.PathLike
    """
    logging.info(f"Converting {file_path} to h5 file.")
    base_path, fname_ext = os.path.split(file_path)
    base_path, _ = os.path.split(base_path)
    fname, _ = os.path.splitext(fname_ext)
    data = psi_dbf_to_dict(file_path)
    h5path = os.path.join(base_path, fname + ".h5")
    dict_to_hdf5(h5path, data)


def convert_dpkg(file_path: os.PathLike):
    """Converts the given dpkg file into a h5 file. The h5 file only contains the necessary information and is zipped with gzip.

    :param file_path: The gpkg file containing PSI data.
    :type file_path: os.PathLike
    """
    logging.info(f"Converting {file_path} to h5 file.")
    base_path, fname_ext = os.path.split(file_path)
    base_path, _ = os.path.split(base_path)
    fname, _ = os.path.splitext(fname_ext)
    data = psi_dbf_to_dict(file_path)
    h5path = os.path.join(base_path, fname + ".h5")
    dict_to_hdf5(h5path, data)


def dict_to_hdf5(
    h5path: os.PathLike,
    data: dict,
    compression: str = "gzip",
    compression_opts: int = 9,
):
    """Saves contents of dictionary into given h5 file. Dates are converted to
    strings following ISO date formatting.

    :param h5path: The path where to save the data.
    :type h5path: os.PathLike
    :param data: The data dictionary to save.
    :type data: dict
    :param compression: Which compression algorithm to use, defaults to "gzip"
    :type compression: str, optional
    :param compression_opts: The compression level, defaults to 9
    :type compression_opts: int, optional
    """
    _, fname = os.path.split(h5path)
    logging.debug(f"Saving data to {fname}.")
    start_time = time.time()
    with h5py.File(h5path, "w") as h5file:
        for k, v in data.items():
            if isinstance(v, dict):
                h5file.create_group(k)
                for k2, v2 in v.items():
                    create_datasets(
                        h5file[k], k2, v2, compression, compression_opts
                    )
            else:
                create_datasets(h5file, k, v, compression, compression_opts)
    end_time = time.time()
    run_time = end_time - start_time
    logging.debug(f"Saving {fname} took {run_time:.3} seconds.")


def create_datasets(
    h5group: h5py.Group, k: str, v, compression: str, compression_opts: int
):
    """Creates a dataset depending on the content of `v`. Used for the nested conversion of a dictionary into a hdf5 file.

    :param h5group: The current group to convert.
    :type h5group: h5py.Group
    :param k: The name of the dataset or group to be created.
    :type k: str
    :param v: The content of the dataset or group.
    :type v: any
    :param compression: Which compression algorithm to use, defaults to "gzip"
    :type compression: str, optional
    :param compression_opts: The compression level, defaults to 9
    :type compression_opts: int, optional
    """
    try:
        if v.dtype == "O":
            v = np.array([val.isoformat().encode() for val in v])
        h5group.create_dataset(
            k,
            data=v,
            compression=compression,
            compression_opts=compression_opts,
        )
    except AttributeError:
        h5group.create_dataset(
            k,
            data=v,
        )
    except TypeError:
        h5group.create_dataset(
            k,
            data=v,
        )


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
            "Converting from this time format is not supported."
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
    if isinstance(time, float):
        t = floatyear_to_datetime(time)
    elif isinstance(time, list):
        t = [get_datetime(n) for n in time]
    elif isinstance(time, np.ndarray):
        t = np.array(get_datetime(list(time)))
    else:
        raise NotImplementedError(
            "Converting from this time format is not supported."
        )
    return t


def datetime_to_floatyear(t: datetime) -> float:
    """Converts a datettime into a floatyear.

    :param time: The timestamp as datetime.
    :type time: datetime
    :return: The floatyear.
    :rtype: float
    """
    return t.year + ((t - datetime(t.year, 1, 1)).days / 365.25)


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

    The output dictionary contains three arrays with the times for the original data `"t"`, the first fit `"t_fit_1"` and the second fit without outliers `"t_fit_2"`. Additionally, the result for each component is saved with its respective key ["U", "E", "W"]. These nested dictionaries contain the original data `"y"`, the first fit `"y_fit_1"` and the second fit without outliers `"y_fit_2"`, each with their corresponding error denoted by the suffix `"_err"`.
    """
    # Dictionary of key names
    comp_names = ["data", "sigm", "dhat", "dres"]
    inv_res = np.reshape(
        results["inversion_results"],
        (3, int(len(results["inversion_results"]) / 3)),
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
