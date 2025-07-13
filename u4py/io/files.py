"""
Contains simple file and folder utilities for u4py. These are mainly wrappers
for tkinter's file dialogs. This is necessary, as sometimes tkinter and
Windows do not interact very well and dialogs are not properly closed, leading
to a crash in windows explorer. This might have been fixed now (?), but for
safety is still included here.

Most functions also can detect if the script is running in a non-interactive
shell without access to a user interface, i.e., on a server. Then the user has
to input the path manually into the command line.
"""

from __future__ import annotations

import os
import pickle as pkl
from tkinter import TclError, Tk, filedialog
from typing import Iterable, Tuple

import geopandas as gp
import shapely
from pathvalidate import sanitize_filename


def get_file_paths(**kwargs) -> list | os.PathLike:
    """Safe wrapper for getting the path of an existing file with a filedialog by tkinter.

    :param kwargs: Keyword arguments supported by :func:`tkinter.filedialog.askopenfilenames` (optional).
    :return: A list of files or a single path if only one file was selected..
    :rtype: list| os.PathLike
    """
    file_list = []
    try:
        root = Tk()
    except TclError:
        print("No display detected.")
        file_path = input("Enter file path:")
        file_list = [
            file_path,
        ]
        if len(file_list) == 1:
            return file_list[0]
        else:
            return file_list
    try:
        root.withdraw()
        file_list = filedialog.askopenfilenames(**kwargs)
    finally:
        root.destroy()
    if len(file_list) == 1:
        return file_list[0]
    return file_list


def get_save_path(**kwargs) -> os.PathLike | str:
    """Safe wrapper for saving a file using filedialog by tkinter.

    :param kwargs: Keyword arguments supported by :func:`tkinter.filedialog.asksaveasfilename` (optional).
    :return: The filepath where to save the data.
    :rtype: os.PathLike | str"""
    try:
        root = Tk()
    except TclError:
        print("No display detected.")
        file_path = input("Enter file path:")
        return file_path
    try:
        root.withdraw()
        file_path = filedialog.asksaveasfilename(**kwargs)
    finally:
        root.destroy()
    return file_path


def get_folder_paths(**kwargs) -> os.PathLike | str:
    """Safe wrapper for to get a folder path using filedialog by tkinter.

    :param kwargs: Keyword arguments supported by :func:`tkinter.filedialog.askdirectory` (optional).
    :return: The filepath to the folder.
    :rtype: os.PathLike | str"""
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


def get_file_list(
    filetype: str,
    folder_path: os.PathLike | str = None,
    recursive: bool = False,
    **kwargs,
) -> list:
    """Asks for folder and returns all files of given filetype

    :param filetype: The filetype to create the list from
    :type filetype: str
    :param folder_path: The base path to look for files, if emtpy the user is asked to select a folder, defaults to None
    :type folder_path: os.PathLike | str, optional
    :param recursive: When True, recurses through all subfolders, defaults to False
    :type recursive: bool, optional
    :return: A list of filepaths to files of the given filetype within the folderpath.
    :rtype: list
    """
    if not folder_path:
        folder_path = get_folder_paths(**kwargs)
    if not recursive:
        file_list = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(filetype)
        ]
    else:
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(filetype):
                    file_list.append(os.path.join(root, filename))
            for dirname in dirs:
                file_list.extend(
                    get_file_list(
                        filetype,
                        os.path.join(root, dirname),
                        recursive=True,
                        **kwargs,
                    )
                )

    return file_list


def multi_split(
    file_path: os.PathLike | str, nsplits: int
) -> os.PathLike | str:
    """Splits the filepath multiple times. Useful for traversing several levels upwards.

    :param file_path: The path to split.
    :type file_path: os.PathLike | str
    :param nsplits: The number of splits to do.
    :type nsplits: int
    :return: The path `nsplits` levels higher.
    :rtype: os.PathLike | str
    """
    for n in range(nsplits):
        file_path = os.path.split(file_path)[0]
    return file_path


def get_rois(
    file_path: os.PathLike | str,
) -> list[Tuple[str, shapely.Polygon]]:
    """Read all regions of interest from the shapefile

    :param file_path: The path to the shapefile containing the regions of interest.
    :type file_path: os.PathLike | str
    :return: A list of (`region name`, `GeoDataFrame`) tuples.
    :rtype: list[Tuple[str, shapely.Polygon]]
    """
    regions = gp.read_file(file_path)
    try:
        rois = [
            (name, geom) for name, geom in zip(regions.Name, regions.geometry)
        ]
    except AttributeError:  # If no `Name` field exists just use numbers
        rois = []
        for ii, geom in enumerate(regions.geometry):
            rois.append((str(ii), geom))
    return rois


def get_all_pickle_data(
    file_path: os.PathLike | str,
) -> list[Tuple[float, float, Iterable]]:
    """Gets pickled data for all stations.

    :param file_path: The file to the pickled data.
    :type file_path: os.PathLike | str
    :return: The data separated into first and second fit.
    :rtype: list[Tuple[float, float, Iterable]]
    """
    with open(file_path, "rb") as pkl_file:
        data = pkl.load(pkl_file)
        fit_1 = [a[0] for a in data if a[0]]
        fit_2 = [a[1] for a in data if a[1]]
        return (fit_1, fit_2)


def get_file_list_tiff(folder_path: os.PathLike | str) -> list[os.PathLike]:
    """Gets a file list of tif files in `TB` folders

    :param folder_path: The path to the folder containing `TB` folders
    :type folder_path: os.PathLike | str
    :return: A list of all tif files.
    :rtype: list
    """

    tb_folders_list = [
        os.path.join(folder_path, fol)
        for fol in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, fol)) and "TB" in fol
    ]
    file_list = []
    for fol in tb_folders_list:
        file_list.extend(
            [
                os.path.join(fol, f)
                for f in os.listdir(fol)
                if f.endswith(".tif")
            ]
        )
    return file_list


def set_point_file_paths(
    source_fpath: os.PathLike | str, name: str, rtype: str
) -> Tuple[os.PathLike, os.PathLike]:
    """Sets the paths for selected psi points.

    :param source_fpath: The file path to the data source.
    :type source_fpath: os.PathLike | str
    :param name: A unique name for the region.
    :type name: str
    :param rtype: The extraction type (e.g., `points`, `clip`, `region`)
    :type rtype: str
    :return: The folder path and file path for data extraction.
    :rtype: Tuple[os.PathLike, os.PathLike]
    """
    base_folder, source_name = os.path.split(source_fpath)
    folder = os.path.join(os.path.split(base_folder)[0], "selected_psi_points")
    os.makedirs(folder, exist_ok=True)
    source = os.path.splitext(source_name)[0]
    fname = sanitize_filename(f"{name}_{rtype}_{source}.shp")
    fpath = os.path.join(folder, fname)
    return folder, fpath


def set_data_file_paths(
    source_fpath: os.PathLike | str, name: str, rtype: str
) -> Tuple[os.PathLike, os.PathLike]:
    """Sets the path for storing the data of selected psi points as a pickle file.

    :param source_fpath: The file path to the data source.
    :type source_fpath: os.PathLike | str
    :param name: A unique name for the region.
    :type name: str
    :param rtype: The extraction type (e.g., `points`, `clip`, `region`)
    :type rtype: str
    :return: The folder path and file path for data extraction.
    :rtype: Tuple[os.PathLike, os.PathLike]
    """
    base_folder, source_name = os.path.split(source_fpath)
    folder = os.path.join(os.path.split(base_folder)[0], "selected_psi_points")
    os.makedirs(folder, exist_ok=True)
    source = os.path.splitext(source_name)[0]
    fname = sanitize_filename(f"{name}_{rtype}_{source}_data.pkl")
    fpath = os.path.join(folder, fname)
    return folder, fpath


def get_file_list_adf(base_folder: os.PathLike | str) -> list[os.PathLike]:
    """Gets a file list of all folders containing individual adf files for a DEM raster dataset.

    :param base_folder: The path to the raster folder containing folders in the form of `g_lat_long`.
    :type base_folder: os.PathLike | str
    :return: The list of paths to the adf files.
    :rtype: list[os.PathLike]
    """

    folders = [
        os.path.join(base_folder, fol)
        for fol in os.listdir(base_folder)
        if fol.startswith("g_")
        and os.path.isdir(os.path.join(base_folder, fol))
    ]
    return folders
