""" Contains simple file and folder utilities for u4py """
import os
import pickle as pkl
from datetime import datetime
from tkinter import TclError, Tk, filedialog
from typing import Tuple

import geopandas as gp
import h5py
import numpy as np

import u4py.analysis.inversion as u4invert
import u4py.analysis.spatial as u4spatial
import u4py.utils.convert as u4convert


def get_file_paths(**kwargs):
    """Safe wrapper for filedialog by tkinter"""
    file_list = []
    try:
        root = Tk()
    except TclError:
        print("No display detected.")
        file_path = input("Enter file path:")
        file_list = [
            file_path,
        ]
        return file_list
    try:
        root.withdraw()
        file_list = filedialog.askopenfilenames(**kwargs)
    finally:
        root.destroy()
    if len(file_list) == 1:
        return file_list[0]
    return file_list


def get_folder_paths(**kwargs):
    """Safe wrapper for filedialog by tkinter"""
    folder_path = ""
    try:
        root = Tk()
    except TclError:
        print("No display detected.")
        folder_path = input("Enter folder path:")
        return folder_path
    try:
        root.withdraw()
        folder_path = filedialog.askdirectory(**kwargs)
    finally:
        root.destroy()

    return folder_path


def get_file_list(filetype=".h5", folder_path=None, **kwargs):
    """Asks for folder and returns all files of given filetype"""
    if not folder_path:
        folder_path = get_folder_paths(**kwargs)
    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(filetype)
    ]
    return file_list


def load_hdf5(file_path, timefmt="datetime", ind=np.array([])) -> dict:
    """Loads data from a hdf5 file. Converts timestamps to datetime.

    Arguments:
        file_path -- The path to the hdf5 file.

    Keyword Arguments:
        timefmt -- Converts timestamps to datetime. (default: {"datetime"})
            Different timestamp formats are supported:
                datetime: Python built-in datetime
                floatyear: Years in float point numbers
        ind -- The indices of the timeseries to load. Creates a subset of the
            data from the file (default: {np.array([])})

    Returns:
        The data as a dictionary.
    """
    with h5py.File(file_path, "r") as h5file:
        data = get_data(h5file, timefmt, ind)
    return data


def load_hdf5_list(file_list: list, timefmt: str = "datetime") -> dict:
    """Loads and merges all data from all files in the file list.

    Arguments:
        file_list -- A list of hdf5 files created from a gpkg file

    Keyword Arguments:
        timefmt -- Timestamp format for 'load_hdf5' (default: {"datetime"})

    Returns:
        The data as a dictionary.
    """
    data = dict()
    for h5path in file_list:
        data_new = load_hdf5(h5path, timefmt)
        if not data:
            data.update(data_new)
        else:
            data = merge_data(data, data_new)

    return data


def merge_data(data: dict, data_new: dict) -> dict:
    """Updates the contents of data depending on the content

    Args:
        data (dict): The current data dictionary
        data_new (dict): The data to be added

    Returns:
        dict: Merged data dictionary
    """

    if data_new["chunk_size"] != data["chunk_size"]:
        raise NotImplementedError(
            "You are attempting to merge two files with different chunk size."
        )
    data["num_points"] += data_new["num_points"]
    data["ps_id"] = np.hstack((data["ps_id"], data_new["ps_id"]))
    data["x"] = np.hstack((data["x"], data_new["x"]))
    data["xmid"] = np.mean(data["x"])
    data["y"] = np.hstack((data["y"], data_new["y"]))
    data["ymid"] = np.mean(data["y"])
    data["z"] = np.hstack((data["z"], data_new["z"]))
    data["timeseries"] = np.vstack(
        (data["timeseries"], data_new["timeseries"])
    )
    return data


def points_to_filelist(gdf: gp.GeoDataFrame, data_folder: os.PathLike) -> list:
    """Gets a list of files to load the data from

    Args:
        gdf (gp.GeoDataFrame): A dataframe containing the points.
        data_folder (os.PathLike): Folder path where a hdf5 file for each
        point in the dataframe is found.

    Returns:
        list: List of all files to be loaded for data aggregation.
    """

    file_list = [
        os.path.join(data_folder, f"PSI_chunk_x{int(p.x)}_y{int(p.y)}.h5")
        for p in gdf.geometry
    ]
    return file_list


def get_data(h5group, timefmt="datetime", ind=np.array([])):
    """
    Recursively gets data from a group. Going deeper if a group is found.
    """
    convert_time = {
        "datetime": datetime.fromisoformat,
        "floatyear": u4convert.get_floatyear,
    }
    data = dict()
    if ind.any():
        ind.sort()
    else:
        ind = None
    for k in h5group.keys():
        if k == "time" or k == "t":
            v = np.array(
                [convert_time[timefmt](val.decode()) for val in h5group[k]]
            )
        else:
            try:
                if ind is None:
                    v = h5group[k][()]
                else:
                    v = h5group[k][ind]
            except ValueError:
                if ind is None:
                    v = h5group[k][()]
                else:
                    v = h5group[k][ind]
            except TypeError:
                v = get_data(h5group[k], timefmt, ind=ind)
        data[k] = v
    return data


def get_data_for_inversion(file_path):
    """Loads file and prepares dataset for inversion"""
    dataset = load_hdf5(file_path, timefmt="floatyear")
    data = u4invert.stack_data(dataset)
    data["sigmE"] = np.ones_like(data["dataE"])
    data["sigmN"] = np.ones_like(data["dataE"])
    data["sigmU"] = np.ones_like(data["dataE"])

    if "inversion_results" in dataset.keys():
        data["inversion_results"] = dataset["inversion_results"]

    return data


def multi_split(file_path: os.PathLike, nsplits: int):
    """Splits the filepath multiple times"""
    for n in range(nsplits):
        file_path = os.path.split(file_path)[0]
    return file_path


def get_select_points_osm(query, psi_file_path, overwrite=False):
    base_folder, source_name = os.path.split(psi_file_path)
    point_files_folder = os.path.join(
        os.path.split(base_folder)[0], "selected_psi_points"
    )
    os.makedirs(point_files_folder, exist_ok=True)
    point_file_name = (
        query["address"] + "_clip_" + os.path.splitext(source_name)[0] + ".shp"
    )
    point_file_path = os.path.join(point_files_folder, point_file_name)

    if os.path.exists(point_file_path) and not overwrite:
        points = gp.GeoDataFrame.from_file(point_file_path)
    else:
        points = u4spatial.select_points_osm(query, psi_file_path)
        points.to_file(point_file_path)
        points = gp.GeoDataFrame.from_file(point_file_path)
    return points


def get_region_points(region, region_name, psi_file_path, overwrite=False):
    base_folder, source_name = os.path.split(psi_file_path)
    point_files_folder = os.path.join(
        os.path.split(base_folder)[0], "selected_psi_points"
    )
    os.makedirs(point_files_folder, exist_ok=True)
    point_file_name = (
        region_name + "_region_" + os.path.splitext(source_name)[0] + ".shp"
    )
    point_file_path = os.path.join(point_files_folder, point_file_name)

    if os.path.exists(point_file_path) and not overwrite:
        points = gp.GeoDataFrame.from_file(point_file_path)
    else:
        points = u4spatial.select_points_region(region, psi_file_path)
        points.to_file(point_file_path)
        points = gp.GeoDataFrame.from_file(point_file_path)
    return points, point_files_folder


def get_point_points(
    point: Tuple[float, float],
    radius: float,
    region_name: str,
    source_file_path: os.PathLike,
    overwrite: bool = False,
) -> gp.GeoDataFrame:
    """Returns the points from `source_file_path` that are within `radius` of
    `point`. Creates a shape file with the extracted points, for quicker
    repeated data access.

    Arguments:
        point -- The point where to start.
        radius -- The search radius around point (in meters)
        region_name -- A sensible name for the extraction point.
        source_file_path -- The file where to extract the data.

    Keyword Arguments:
        overwrite -- Whether to overwrite the shape file (default: {False})

    Returns:
        A GeoDataFrame with the points including all relevant data.
    """
    base_folder, source_name = os.path.split(source_file_path)
    output_folder = os.path.join(
        os.path.split(base_folder)[0], "selected_psi_points"
    )
    os.makedirs(output_folder, exist_ok=True)
    output_file_name = (
        region_name + "_points_" + os.path.splitext(source_name)[0] + ".shp"
    )
    output_file_path = os.path.join(output_folder, output_file_name)

    if os.path.exists(output_file_path) and not overwrite:
        points = gp.GeoDataFrame.from_file(output_file_path)
    else:
        points = u4spatial.select_points_point(point, radius, source_file_path)
        points.to_file(output_file_path)
        points = gp.GeoDataFrame.from_file(output_file_path)
    return points


def get_pickled_inversion_results(
    file_path: os.PathLike, ind: slice = None, points: gp.GeoDataFrame = None
) -> Tuple[list[tuple], float]:
    """Loads data from selected pickle file and also returns avg. distance to
    nearest inversion results.

    Use only one of the selection criteria:
        - ind: Uses the indices in the list to slice the data.
        - points: Does a spatial search in the data to extract the data at the
        coordinates of points.

    Arguments:
        file_path -- The file path of the pickle file

    Keyword Arguments:
        ind -- Indices where to look for data. If None takes all data. (default: {None})
        points -- List of points where to look for data. If None takes all data. (default: {None})

    Returns:
        data (selection) and mean distance to data (only for point selection)
    """
    with open(file_path, "rb") as pklfile:
        data = pkl.load(pklfile)[0]

    if (ind is not None) and (points is not None):
        ValueError("Two selections applied, either use ind or points!")
    elif ind is not None:
        if isinstance(ind, list):
            return [data[ii] for ii in ind], 0
        elif isinstance(ind, int):
            return [data[ind]], 0
    elif points is not None:
        closest = u4spatial.spatial_lookup(data, points)
        data = [data[c[1]] for c in closest]
        distance = np.mean([c[0] for c in closest])
        return data, distance
    else:
        return data, 0


def load_data_from_points(
    input_path: os.PathLike, points: gp.GeoDataFrame
) -> dict:
    """Gets the data at the specified points.

    Arguments:
        input_path -- The path to a hdf5 file or folder with files to load data.
        points -- The points where to query the data.

    Returns:
        A dictionary containing the data.
    """
    # Loading for a single file:
    if input_path.endswith(".h5"):
        ind = points.source_ind.to_numpy()
        data = load_hdf5(input_path, ind=ind)

    # Loading for a folder of split files (better parallelization)
    else:
        data_filelist = points_to_filelist(points, input_path)
        data = load_hdf5_list(data_filelist)
    return data
