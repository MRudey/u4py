""" Contains simple file and folder utilities for u4py """

import os
from datetime import datetime
from tkinter import Tk, filedialog

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
        root.withdraw()
        file_list = filedialog.askopenfilenames(**kwargs)
    finally:
        root.destroy()

    return file_list


def get_folder_paths(**kwargs):
    """Safe wrapper for filedialog by tkinter"""
    folder_path = ""
    try:
        root = Tk()
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


def load_hdf5(file_path, timefmt="datetime", ind=None):
    """
    Loads data from a hdf5 file. Converts timestamps to datetime.
    Different timestamp formats are supported:
        datetime: Python built-in datetime
        floatyear: Years in float point numbers
    """
    with h5py.File(file_path, "r") as h5file:
        data = get_data(h5file, timefmt, ind)
    return data


def get_data(h5group, timefmt="datetime", ind=None):
    """
    Recursively gets data from a group. Going deeper if a group is found.
    """
    convert_time = {
        "datetime": datetime.fromisoformat,
        "floatyear": u4convert.get_floatyear,
    }
    data = dict()
    ind.sort()
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


def get_select_points(query, psi_file_path, overwrite=False):

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
        points = u4spatial.select_points(query, psi_file_path)
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
    return points


def get_point_points(
    point, radius, region_name, psi_file_path, overwrite=False
):

    base_folder, source_name = os.path.split(psi_file_path)
    point_files_folder = os.path.join(
        os.path.split(base_folder)[0], "selected_psi_points"
    )
    os.makedirs(point_files_folder, exist_ok=True)
    point_file_name = (
        region_name + "_points_" + os.path.splitext(source_name)[0] + ".shp"
    )
    point_file_path = os.path.join(point_files_folder, point_file_name)

    if os.path.exists(point_file_path) and not overwrite:
        points = gp.GeoDataFrame.from_file(point_file_path)
    else:
        points = u4spatial.select_points_point(point, radius, psi_file_path)
        points.to_file(point_file_path)
        points = gp.GeoDataFrame.from_file(point_file_path)
    return points
