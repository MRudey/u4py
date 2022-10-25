""" Contains functions for file conversion """
import os
from datetime import datetime, timedelta

import geopandas
import h5py
import numpy as np
import shapely.geometry as shpgeo
import u4py.utils.files as u4files
from dbfread import DBF
from tqdm import tqdm


def dbf_to_dict(file_in: os.PathLike):
    """Generic conversion of dbf file to dictionary"""
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


def psi_dbf_to_dict(file_in: os.PathLike):
    """
    Takes a dbf file and returns a dictionary with numpy arrays for x, y, z
    coordinates, PS_ID and timeseries for each entry.
    """
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
            key_list = [
                k for k in dbffile.field_names if k not in non_time_keys
            ]
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
            "var_mean_vel": mean_vel,
        }
    return output


def key_to_time(
    key,
    starttime=datetime(year=2016, month=4, day=1),
    startindex=20150,
    acqdiff=timedelta(days=6),
):
    """Converts a key timestamp to a datetime

    Args:
        key (str):
            Timestamp key
        starttime (datetime, optional):
            Time when first PSI data was taken. Defaults to datetime(2016,4,1).
        startindex (int, optional):
            First index when counting is started. Defaults to 20150.
        acqdiff (timedelta, optional):
            Timedifference between acquisitions. Defaults to timedelta(days=6).

    Returns:
        datetime: converted and referenced datetime
    """
    timediff = (int(key[5:]) - startindex) * acqdiff
    return starttime + timediff


def chunk_data(data, save_folder, chunksize=1000, min_values=5, compress=True):
    """
    Chunks data into many smaller files with spanning a square of `chunksize`
    meters. Discards chunks with less than `min_values`.
    """
    xrange = get_bounds(np.min(data["x"]), np.max(data["x"]), chunksize)
    yrange = get_bounds(np.min(data["y"]), np.max(data["y"]), chunksize)
    chunk_list = get_chunks(xrange, yrange)
    os.makedirs(save_folder, exist_ok=True)
    for chunk in tqdm(chunk_list, desc="Chunking data", leave=False):
        xlow = data["x"] >= chunk[0]
        xhigh = data["x"] < chunk[1]
        ylow = data["y"] >= chunk[2]
        yhigh = data["y"] < chunk[3]
        slc = np.nonzero(xlow * xhigh * ylow * yhigh)

        if len(slc[0]) > min_values:
            chunk_name = "PSI_chunk_x%i_y%i.h5" % (chunk[0], chunk[2])
            output = dict()
            for k in data.keys():
                if k == "time":
                    output[k] = data[k]
                else:
                    output[k] = data[k][slc]
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


def get_bounds(minval, maxval, chunksize):
    """
    Returns boundaries for given chunksize
    """
    minbound = int(np.floor(minval / chunksize) * chunksize)
    maxbound = int(np.ceil(maxval / chunksize) * chunksize)
    return np.arange(minbound, maxbound, chunksize)


def get_chunks(xrange, yrange):
    """
    Returns the corners for chunking
    """
    chunk_list = []
    for ii, xm in enumerate(xrange):
        if ii < len(xrange) - 1:
            for jj, ym in enumerate(yrange):
                if jj < len(yrange) - 1:
                    chunk_list.append([xm, xrange[ii + 1], ym, yrange[jj + 1]])
    return chunk_list


def convert_file(file_path):
    """
    Converts the given dbf file into a h5 file. The h5 file only contains the
    necessary information and is zipped with gzip.
    """
    base_path, fname_ext = os.path.split(file_path)
    base_path, _ = os.path.split(base_path)
    fname, _ = os.path.splitext(fname_ext)
    data = psi_dbf_to_dict(file_path)
    h5path = os.path.join(base_path, fname + ".h5")
    dict_to_hdf5(h5path, data)


def dict_to_hdf5(
    h5path: os.PathLike, data: dict, compression="gzip", compression_opts=9
):
    """
    Saves contents of dictionary into given h5 file. Dates are converted to
    strings following ISO date formatting.
    """
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


def create_datasets(h5group, k, v, compression, compression_opts):
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


def merge_data(inputs):
    """Converts average LOS movement into EW and UD component

    Args:
        base_path (os.PathLike): Path to folder with ASCE and DESC folder
        file_name (os.PathLike): Common file for calculation
    """
    base_path = inputs[0]
    file_name = inputs[1]
    path_a = os.path.join(base_path, "BBD_EW", file_name)
    path_d = os.path.join(base_path, "BBD_Vert", file_name)
    output_path = os.path.join(base_path, "merged")
    data_ew = u4files.load_hdf5(path_a)
    data_ud = u4files.load_hdf5(path_d)

    common_ps_id = np.nonzero(data_ew["ps_id"] == data_ud["ps_id"])
    output_data = dict()
    for ii in common_ps_id[0]:
        station = data_ew["ps_id"][ii]
        output_data[f"{station}"] = {
            "t": data_ew["time"],
            "dataE": data_ew["timeseries"][ii],
            "dataN": data_ew["timeseries"][ii],
            "dataU": data_ud["timeseries"][ii],
            # "sigmE": sigmE,
            # "sigmN": sigmE,
            # "sigmU": sigmE,
            "station": f"{station}",
            "xmid": data_ew["xmid"],
            "ymid": data_ew["ymid"],
            "chunk_size": data_ew["chunk_size"],
        }
    dict_to_hdf5(os.path.join(output_path, file_name), output_data)
