"""
Contains simple file and folder utilities for u4py. These are mainly wrappers for tkinter's file dialogs. This is necessary, as sometimes tkinter and Windows do not interact very well and dialogs are not properly closed, leading to a crash in windows explorer. This might have been fixed now (?), but for safety is still included here.

Most functions also can detect if the script is running in a non-interactive shell without access to a user interface, i.e., on a server. Then the user has to input the path manually into the command line.
"""
from __future__ import annotations

import configparser
import logging
import os
import pickle as pkl
from datetime import datetime
from multiprocessing import Pool
from tkinter import TclError, Tk, filedialog
from typing import Iterable, Tuple

import fiona
import geopandas as gp
import h5py
import numpy as np
import osmnx
import pandas as pd
import rasterio as rio
import shapely
import shapely.geometry as shpgeo
import shapely.ops as shpops
from rasterio.mask import mask as riomask
from rasterio.merge import merge as riomerge
from rasterio.transform import Affine
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.analysis.spatial as u4spatial
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert


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
        return file_list
    try:
        root.withdraw()
        file_list = filedialog.askopenfilenames(**kwargs)
    finally:
        root.destroy()
    if len(file_list) == 1:
        return file_list[0]
    return file_list


def get_save_path(**kwargs) -> os.PathLike:
    """Safe wrapper for saving a file using filedialog by tkinter.

    :param kwargs: Keyword arguments supported by :func:`tkinter.filedialog.asksaveasfilename` (optional).
    :return: The filepath where to save the data.
    :rtype: os.PathLike"""
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


def get_folder_paths(**kwargs) -> os.PathLike:
    """Safe wrapper for to get a folder path using filedialog by tkinter.

    :param kwargs: Keyword arguments supported by :func:`tkinter.filedialog.askdirectory` (optional).
    :return: The filepath to the folder.
    :rtype: os.PathLike"""
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
    filetype: str = ".h5",
    folder_path: os.PathLike = None,
    recursive: bool = False,
    **kwargs,
) -> list:
    """Asks for folder and returns all files of given filetype

    :param filetype: The filetype to create the list from, defaults to ".h5"
    :type filetype: str, optional
    :param folder_path: The base path to look for files, if emtpy the user is asked to select a folder, defaults to None
    :type folder_path: os.PathLike, optional
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


def load_hdf5(
    file_path: os.PathLike,
    timefmt: str = "datetime",
    ind: np.ndarray = np.array([]),
) -> dict:
    """Loads data from a hdf5 file. Converts timestamps to datetime.

    :param file_path:  The path to the hdf5 file.
    :type file_path: os.PathLike
    :param timefmt: Converts timestamps to datetime. Different timestamp formats are supported: `"datetime"` - Python built-in datetime, `"floatyear"` - Years in float point numbers, defaults to "datetime"
    :type timefmt: str, optional
    :param ind: The indices of the timeseries to load. Creates a subset of the data from the file, defaults to np.array([])
    :type ind: np.ndarray, optional
    :return: The data as a dictionary.
    :rtype: dict
    """
    with h5py.File(file_path, "r") as h5file:
        data = get_data(h5file, timefmt, ind)
    return data


def load_hdf5_list(file_list: list, timefmt: str = "datetime") -> dict:
    """Loads and merges all data from all files in the file list.

    :param file_list: A list of hdf5 files created from a gpkg file
    :type file_list: list
    :param timefmt: Timestamp format for 'load_hdf5', defaults to "datetime"
    :type timefmt: str, optional
    :return: The data as a dictionary.
    :rtype: dict
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

    :param data: The current data dictionary
    :type data: dict
    :param data_new: The data to be added
    :type data_new: dict
    :raises NotImplementedError: Raised when files with different chunk sizes are merged.
    :return: Merged data dictionary
    :rtype: dict
    """
    try:
        # Type 1 (xyz notation)
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
    except KeyError:
        # Type 2 (dataE, dataN, dataU notation)
        data.update(data_new)
        data["inversion_results"] = None
    return data


def merge_directions(inputs: Tuple[os.PathLike, os.PathLike]):
    """Converts average LOS movement into EW and UD component

    :param inputs: A tuple containing the pat to folder with ASCE and DESC folder and the common filename for calculation.
    :type inputs: Tuple[os.PathLike, os.PathLike]
    """
    base_path = inputs[0]
    file_name = inputs[1]
    logging.info(f"Merging {file_name}")
    path_a = os.path.join(base_path, "BBD_EW", file_name)
    path_d = os.path.join(base_path, "BBD_Vert", file_name)
    output_path = os.path.join(base_path, "merged")
    data_ew = load_hdf5(path_a)
    data_ud = load_hdf5(path_d)

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
    u4convert.dict_to_hdf5(os.path.join(output_path, file_name), output_data)


def points_to_filelist(gdf: gp.GeoDataFrame, data_folder: os.PathLike) -> list:
    """Gets a list of files to load the data from

    :param gdf: A dataframe containing the points.
    :type gdf: gp.GeoDataFrame
    :param data_folder: Folder path where a hdf5 file for each point in the dataframe is found.
    :type data_folder: os.PathLike
    :return: List of all files to be loaded for data aggregation.
    :rtype: list
    """

    file_list = [
        os.path.join(data_folder, f"PSI_chunk_x{int(p.x)}_y{int(p.y)}.h5")
        for p in gdf.geometry
    ]
    return file_list


def get_data(
    h5group: h5py.Group,
    timefmt: str = "datetime",
    ind: np.ndarray = np.array([]),
) -> dict:
    """

    :param h5group: Recursively gets data from a group. Going deeper if a group is found.
    :type h5group: h5py.Group
    :param timefmt: The format for time to use, defaults to "datetime"
    :type timefmt: str, optional
    :param ind: Only loads the data at the indices `ind`, defaults to np.array([])
    :type ind: np.ndarray, optional
    :return: The loaded data.
    :rtype: dict
    """
    convert_time = {
        "datetime": datetime.fromisoformat,
        "floatyear": u4convert.get_floatyear,
    }
    data = dict()
    if ind.size > 0:
        ind.sort()
    for k in h5group.keys():
        if k == "time" or k == "t":
            v = np.array(
                [convert_time[timefmt](val.decode()) for val in h5group[k]]
            )
        else:
            try:
                if ind.size > 0:
                    v = h5group[k][ind]
                else:
                    v = h5group[k][()]
            except ValueError:
                if ind.size > 0:
                    v = h5group[k][ind]
                else:
                    v = h5group[k][()]
            except TypeError:
                v = get_data(h5group[k], timefmt, ind=ind)
        data[k] = v
    return data


def get_data_for_inversion(file_path: os.PathLike) -> dict:
    """Loads file and prepares dataset for inversion.

    :param file_path: The path to the data file.
    :type file_path: os.PathLike
    :return: A dictionary formatted for inversion
    :rtype: dict
    """
    dataset = load_hdf5(file_path, timefmt="floatyear")
    data = u4invert.reformat_dict(dataset)
    return data


def multi_split(file_path: os.PathLike, nsplits: int) -> os.PathLike:
    """Splits the filepath multiple times. Useful for traversing several levels upwards.

    :param file_path: The path to split.
    :type file_path: os.PathLike
    :param nsplits: The number of splits to do.
    :type nsplits: int
    :return: The path `nsplits` levels higher.
    :rtype: os.PathLike
    """
    for n in range(nsplits):
        file_path = os.path.split(file_path)[0]
    return file_path


def get_osm_points(
    query: dict, source_file_path: os.PathLike, overwrite: bool = False
) -> Tuple[gp.GeoDataFrame, os.PathLike]:
    """Gets the PSI points in a region defined by an OSM query and saves them
    into a shape file for faster access.

    :param query: The OSM query
    :type query: dict
    :param source_file_path: Path to folder or file where the PSI data is found.
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the output shape file, defaults to False
    :type overwrite: bool, optional
    :return: Returns the points as GeoDataFrame with time series attached and the path where the folder is found.
    :rtype: Tuple[gp.GeoDataFrame, os.PathLike]
    """
    base_folder, source_name = os.path.split(source_file_path)
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
        points = u4spatial._select_points_osm(query, source_file_path)
        points.to_file(point_file_path)
        points = gp.GeoDataFrame.from_file(point_file_path)
    return points


def get_region_points(
    region: gp.GeoDataFrame,
    region_name: str,
    source_file_path: os.PathLike,
    overwrite: bool = False,
    crs: str = "EPSG:32632",
) -> Tuple[gp.GeoDataFrame, os.PathLike]:
    """Gets the PSI points in a region and saves them into a shape file for
    faster access.

    :param region: A `GeoDataFrame` of the region, e.g. from a shape file
    :type region: gp.GeoDataFrame
    :param region_name: The name of the region.
    :type region_name: str
    :param source_file_path: Path to folder or file where the PSI data is found.
    :type source_file_path: os.PathLike
    :param overwrite:  Whether to overwrite the output shape file, defaults to False
    :type overwrite: bool, optional
    :param crs: The CRS of the input shapes, defaults to "EPSG:32632"
    :type crs: str, optional
    :return: Returns the points as GeoDataFrame with time series attached and the path where the folder is found.
    :rtype: Tuple[gp.GeoDataFrame, os.PathLike]
    """
    base_folder, source_name = os.path.split(source_file_path)
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
        if isinstance(region, gp.GeoDataFrame):
            points = u4spatial._select_points_region(
                region, source_file_path, crs=crs
            )
        elif isinstance(region, pd.Series):
            points = u4spatial._select_points_region(
                region, source_file_path, crs=crs
            )
        elif isinstance(region, shapely.Polygon):
            region_gdf = gp.GeoDataFrame(
                geometry=gp.GeoSeries(region), crs=crs
            )
            points = u4spatial._select_points_region(
                region_gdf, source_file_path, crs=crs
            )
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

    :param point: The point where to start.
    :type point: Tuple[float, float]
    :param radius: The search radius around point (in meters).
    :type radius: float
    :param region_name: A sensible name for the extraction point.
    :type region_name: str
    :param source_file_path: The file where to extract the data.
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the shape file, defaults to False
    :type overwrite: bool, optional
    :return: A GeoDataFrame with the points including all relevant data.
    :rtype: gp.GeoDataFrame
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
        points = u4spatial._select_points_point(
            point, radius, source_file_path
        )
        points.to_file(output_file_path)
        points = gp.GeoDataFrame.from_file(output_file_path)
    return points


def get_pickled_inversion_results(
    file_path: os.PathLike, ind: slice = None, points: gp.GeoDataFrame = None
) -> Tuple[list[tuple], float]:
    """Loads data from selected pickle file and also returns avg. distance to
    nearest inversion results.

    :param file_path: The file path of the pickle file
    :type file_path: os.PathLike
    :param ind: Indices where to look for data. If None takes all data, defaults to None
    :type ind: slice, optional
    :param points: List of points where to look for data. If None takes all data, defaults to None
    :type points: gp.GeoDataFrame, optional
    :return: data (selection) and mean distance to data (only for point selection)
    :rtype: Tuple[list[tuple], float]

    Use only one of the selection criteria:
        - ind: Uses the indices in the list to slice the data.
        - points: Does a spatial search in the data to extract the data at the coordinates of points.
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
        closest = u4spatial._spatial_lookup(data, points)
        data = [data[c[1]] for c in closest]
        distance = np.mean([c[0] for c in closest])
        return data, distance
    else:
        return data, 0


def load_data_from_points(
    input_path: os.PathLike, points: gp.GeoDataFrame
) -> dict:
    """Loads the data from a h5 file or folder at the locations extracted from points.

    :param h5path: The path to the data.
    :type h5path: os.PathLike
    :param points: A point shape file loaded from disk or generated by :func:`get_point_points`
    :type points: gp.GeoDataFrame
    :return: The data as a merged, single dictionary.
    :rtype: dict
    """
    logging.info("Loading data from points.")
    # Loading for a single file:
    if input_path.endswith(".h5"):
        ind = points.source_ind.to_numpy()
        data = load_hdf5(input_path, ind=ind)

    # Loading for a folder of split files (better parallelization)
    else:
        data_filelist = points_to_filelist(points, input_path)
        data = load_hdf5_list(data_filelist)
    return data


def ndarray_to_geotiff(
    Z: np.ndarray, bounds: tuple, file_path: os.PathLike, crs: str
):
    """Saves a numpy nd array and its bounds to a georeferenced tiff.

    :param Z: The array
    :type Z: np.ndarray
    :param bounds: The edges of the array in real world coordinates
    :type bounds: tuple
    :param file_path: The path to save the tiff to.
    :type file_path: os.PathLike
    :param crs: The coordinate system to use.
    :type crs: str
    """
    height, width = Z.shape
    res = (bounds[1] - bounds[0]) / width
    transform = Affine.translation(bounds[0] - res, bounds[2]) * Affine.scale(
        res, res
    )

    with rio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=Z.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(Z, 1)


def load_pickled_results(pickle_path: os.PathLike) -> Tuple[dict, int]:
    """Loads the given pickle file containing the inversion results for plotting.

    :param pickle_path: The path to the pickle file.
    :type pickle_path: os.PathLike
    :return: A results dictionary and chunk size in a tuple.
    :rtype: Tuple[dict, int]
    """
    """"""
    with open(pickle_path, "rb") as pkl_file:
        results, chunk_size = pkl.load(pkl_file)
    return results, chunk_size


def load_tiff(tiff_file_path: os.PathLike) -> rio.DatasetReader:
    """Loads a tiff file for plotting with rasterio.

    :param tiff_file_path: The path to the georeference tiff file.
    :type tiff_file_path: os.PathLike
    :return: The openend dataset.
    :rtype: rio.DatasetReader
    """
    tiff_tile = rio.open(tiff_file_path)
    return tiff_tile


def get_all_tiff_regions(
    project: configparser.ConfigParser, overwrite: bool = False
) -> gp.GeoDataFrame:
    """Gets the tiff files as regions (WIP)

    :param project: The project config.
    :type project: configparser.ConfigParser
    :param overwrite: Overwrite existing files if True, defaults to False
    :type overwrite: bool, optional
    :return: The tiffs as a geodataframe.
    :rtype: gp.GeoDataFrame
    """
    all_tiff_path = os.path.join(
        project["paths"]["diff_plan_path"], "all_tiff_overviews.shp"
    )
    if os.path.exists(all_tiff_path) and not overwrite:
        all_tiff_gdf = gp.GeoDataFrame.from_file(all_tiff_path)
    else:
        tiff_file_list = get_file_list(
            folder_path=project["paths"]["diff_plan_path"],
            filetype=".tif",
            recursive=True,
        )
        all_tiff_list = []
        for tiff_file in tqdm(tiff_file_list, desc="Generating overviews"):
            tile = load_tiff(tiff_file)
            all_tiff_list.append(u4spatial.bounds_to_polygon(tile.bounds))
        all_tiff_gdf = gp.GeoDataFrame(
            {"src_path": tiff_file_list, "geometry": all_tiff_list},
            crs=tile.crs,
        )
        all_tiff_gdf.to_file(all_tiff_path)
    return all_tiff_gdf


def get_rois(file_path: os.PathLike) -> list[Tuple[str, shapely.Polygon]]:
    """Read all regions of interest from the shapefile

    :param file_path: The path to the shapefile containing the regions of interest.
    :type file_path: os.PathLike
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
    file_path: os.PathLike,
) -> list[Tuple[float, float, Iterable]]:
    """Gets pickled data for all stations.

    :param file_path: The file to the pickled data.
    :type file_path: os.PathLike
    :return: The data separated into first and second fit.
    :rtype: list[Tuple[float, float, Iterable]]
    """
    with open(file_path, "rb") as pkl_file:
        data = pkl.load(pkl_file)
        fit_1 = [a[0] for a in data if a[0]]
        fit_2 = [a[1] for a in data if a[1]]
        return (fit_1, fit_2)


def get_file_list_tiff(folder_path: os.PathLike) -> list:
    """Gets a file list of tif files in `TB` folders

    :param folder_path: The path to the folder containing `TB` folders
    :type folder_path: os.PathLike
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


def get_osm_tiff(
    query: dict, source_file_path: os.PathLike, overwrite: bool = False
) -> list[os.PathLike]:
    """Gets a list of paths to tiff files within a specified OSM query.

    :param query: The OSM query
    :type query: dict
    :param source_file_path: The path to the folder where the tiffs are stored (in `TB` folders)
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the output shapefile, defaults to False
    :type overwrite: bool, optional
    :return: A list of paths to the tiff files within the osm region.
    :rtype: list[os.PathLike]
    """
    file_list = get_file_list_tiff(source_file_path)
    points = u4spatial._select_points_osm(query, file_list)
    out_file_list = np.array(file_list)[points.source_index]
    return np.unique(out_file_list).tolist()


def get_region_tiff(
    region: gp.GeoDataFrame,
    region_name: str,
    source_file_path: os.PathLike,
    overwrite: bool = False,
    crs: str = "EPSG:32632",
) -> list[os.PathLike]:
    """Gets a list of paths to tiff files in a region.

    :param region: A `GeoDataFrame` of the region, e.g. from a shape file
    :type region: gp.GeoDataFrame
    :param region_name: The name of the region.
    :type region_name: str
    :param source_file_path: The path to the folder where the tiffs are stored (in `TB` folders)
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the output shapefile, defaults to False
    :type overwrite: bool, optional
    :param crs: The CRS of the input shapes, defaults to "EPSG:32632"
    :type crs: str, optional
    :return: A list of paths to the tiff files within the osm region.
    :rtype: list[os.PathLike]
    """
    logging.info("Getting tiffs from region")

    file_list = get_file_list_tiff(source_file_path)
    if isinstance(region, gp.GeoDataFrame):
        points = u4spatial._select_points_region(region, file_list, crs=crs)
    elif isinstance(region, pd.Series):
        points = u4spatial._select_points_region(region, file_list, crs=crs)
    elif isinstance(region, shapely.Polygon):
        region_gdf = gp.GeoDataFrame(geometry=gp.GeoSeries(region), crs=crs)
        points = u4spatial._select_points_region(
            region_gdf, file_list, crs=crs
        )
    out_file_list = np.array(file_list)[points.source_index]
    return np.unique(out_file_list).tolist()


def get_point_tiff(
    point: Tuple[float, float],
    radius: float,
    region_name: str,
    source_file_path: os.PathLike,
    overwrite: bool = False,
) -> list[os.PathLike]:
    """Returns paths of tiff files that are within `radius` of `point`.

    :param point: The point where to start.
    :type point: Tuple[float, float]
    :param radius: The search radius around point (in meters).
    :type radius: float
    :param region_name: A sensible name for the extraction point.
    :type region_name: str
    :param source_file_path: The path to the folder where the tiffs are stored (in `TB` folders)
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the output shapefile, defaults to False
    :type overwrite: bool, optional
    :return: A list of paths to the tiff files within the osm region.
    :rtype: list[os.PathLike]
    """
    logging.info("Getting tiffs around point")

    file_list = get_file_list_tiff(source_file_path)
    points = u4spatial._select_points_point(point, radius, file_list)
    out_file_list = np.array(file_list)[points.source_index]
    return np.unique(out_file_list).tolist()


def get_clipped_shapefile(
    file_path_in: os.PathLike,
    mask: gp.GeoDataFrame | gp.GeoSeries,
    region_name: str,
    fclass: list[str] = [],
    overwrite: bool = False,
) -> Tuple[gp.GeoDataFrame, os.PathLike]:
    """Reads the contents of a shapefile and returns it clipped by `mask`

    :param file_path_in: The path to the shapefile
    :type file_path_in: os.PathLike
    :param mask: The mask used for clipping
    :type mask: gp.GeoDataFrame | gp.GeoSeries
    :param fclass: A list of feature classes to use from the original dataset, defaults to [] (all `fclasses`)
    :type fclass: list, optional
    :param overwrite: Whether to overwrite the existing shapefiles, defaults to False
    :type overwrite: bool, optional
    :return: The clipped dataset and the path to the output file.
    :rtype: Tuple[gp.GeoDataFrame, os.PathLike]
    """
    logging.info("Getting clipped shapefile")

    base_path, file_name = os.path.split(file_path_in)
    fname, _ = os.path.splitext(file_name)
    clipped_base = os.path.join(base_path, "clipped_shapes")
    os.makedirs(clipped_base, exist_ok=True)

    region_name = region_name.replace(" ", "")
    clipped_path = os.path.join(clipped_base, fname + region_name + ".shp")

    if os.path.exists(clipped_path) and not overwrite:
        logging.info("Reading from existing shapefile.")
        clipped_data = fiona_load(clipped_path)
    else:
        logging.info("Creating new clipped shapefile")
        in_data = fiona_load(file_path_in, fclasses=fclass)
        logging.info("Clipping Data")
        if in_data.crs != mask.crs:
            mask = mask.to_crs(in_data.crs)
        clipped_data = in_data.clip(mask)
        logging.info("Saving clipped data.")
        clipped_data.to_file(clipped_path)
    return clipped_data, clipped_path


def get_buffered_shapefiles(
    places_path: os.PathLike,
    mask: gp.GeoDataFrame | gp.GeoSeries,
    region_name: str,
    shp_cfg: dict,
    out_crs: str = "",
    overwrite: bool = False,
) -> Tuple[gp.GeoDataFrame, os.PathLike]:
    """Gets buffered and masked shapes for clipping from shapefile, recreates the shapefile if not found.

    :param places_path: The path to the folder containing the shapefiles.
    :type places_path: os.PathLike
    :param mask: The mask for the dataset (e.g. a region of interest)
    :type mask: gp.GeoDataFrame | gp.GeoSeries
    :param shp_cfg: The configuration dictionary for the shapefiles in `places_path`
    :type shp_cfg: dict
    :param overwrite: Whether to overwrite the final file, defaults to False
    :type overwrite: bool, optional
    :return: A geodataframe with the shapes and the path to the output file.
    :rtype: Tuple[gp.GeoDataFrame, os.PathLike]
    """
    logging.info("Getting buffered shapefiles")
    merged_base = os.path.join(places_path, "buffered_and_merged_shapes")
    os.makedirs(merged_base, exist_ok=True)

    merged_path = os.path.join(merged_base, region_name + "_merged.shp")

    if os.path.exists(merged_path) and not overwrite:
        logging.info("Loading from existing merged and buffered file")
        merged_gdf = gp.read_file(merged_path)
    else:
        logging.info("Creating new merged and clipped shapefiles")
        shp_data = dict()
        for osm_type in tqdm(
            shp_cfg["shp_file"].keys(),
            desc="Getting Clipped Shapefiles",
            leave=False,
        ):
            shp_data[osm_type], _ = get_clipped_shapefile(
                os.path.join(
                    places_path,
                    shp_cfg["shp_file"][osm_type],
                ),
                mask,
                region_name,
                fclass=shp_cfg["fclass"][osm_type],
                overwrite=overwrite,
            )

        logging.info("Merging and buffering")
        for ii, kk in enumerate(shp_data.keys()):
            if ii == 0:
                first_crs = shp_data[kk].crs
            if shp_data[kk].crs != first_crs:
                shp_data[kk] = shp_data[kk].to_crs(first_crs)

        merged_geometry = u4spatial.buffer_and_merge(shp_data, shp_cfg)
        logging.info("Saving Merged Geometry")
        merged_gdf = gp.GeoDataFrame(
            geometry=merged_geometry, crs=shp_data[osm_type].crs
        )
        if out_crs:
            logging.debug("Converting to different CRS")
            merged_gdf = merged_gdf.to_crs(out_crs)
        merged_gdf.to_file(merged_path)
    return merged_gdf, merged_path


def get_clipped_tiff_list(
    tiff_file_list: list[os.PathLike],
    mask_shp: os.PathLike,
    region_name: str,
    overwrite: bool = False,
) -> list[os.PathLike]:
    """Gets all tiffs from `tiff_file_list` clipped by the shapes in
    `mask_shp`. The results are saved in a separate folder for quicker loading.

    :param tiff_file_list: The file list of tiffs to clip
    :type tiff_file_list: list[os.PathLike]
    :param mask_shp: The shapefile containing the geometries for clipping.
    :type mask_shp: os.PathLike
    :param region_name: The name of the region for naming the output folder.
    :type region_name: str
    :param overwrite: Whether to overwrite the existing results, defaults to False
    :type overwrite: bool, optional
    :return: A list with paths to the clipped tiff files.
    :rtype: list[os.PathLike]
    """
    logging.info("Getting clipped tiffs")
    base_path = multi_split(tiff_file_list[0], 2)
    ctiff_fol = os.path.join(base_path, f"clipped_tiffs_{region_name}")
    os.makedirs(ctiff_fol, exist_ok=True)

    clipped_tiff_list = []
    if os.path.exists(ctiff_fol) and not overwrite:
        logging.info("Loading existing data")
        clipped_tiff_list = [
            os.path.join(ctiff_fol, tf)
            for tf in os.listdir(ctiff_fol)
            if tf.endswith(".tif")
        ]
    if not clipped_tiff_list:
        logging.info("Clipping tiff files.")
        with fiona.open(mask_shp, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        args = [(fp, ctiff_fol, shapes) for fp in tiff_file_list]
        with Pool(u4config.cpu_count) as p:
            logging.info("Starting Parallel Pool")
            list(
                tqdm(
                    p.imap_unordered(batch_clip_tiff, args),
                    total=len(tiff_file_list),
                    desc="Masking Rasters",
                    leave=False,
                )
            )
        clipped_tiff_list = [
            os.path.join(ctiff_fol, tf)
            for tf in os.listdir(ctiff_fol)
            if tf.endswith(".tif")
        ]
    return clipped_tiff_list


def batch_clip_tiff(args):
    """Wrapper for clipping with parallel Pool

    :param args: The arguments
    """
    clip_tiff(*args)


def clip_tiff(in_path: os.PathLike, ctiff_fol: os.PathLike, shapes: list):
    """Clips a tiff file with `shapes` and saves it to `ctiff_fol`.

    :param in_path: The path of the tif file to clip.
    :type in_path: os.PathLike
    :param ctiff_fol: The folder where to store the clipped tiffs.
    :type ctiff_fol: os.PathLike
    :param shapes: The shapes with which to clip
    :type shapes: list
    """
    logging.debug(f"Clipping {in_path}")
    _, fname = os.path.split(in_path)
    out_path = os.path.join(ctiff_fol, fname)
    with rio.open(in_path, "r") as src:
        out_image, out_transform = riomask(src, shapes, invert=True)
        out_meta = src.meta
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    with rio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)


def get_merged_tiff_path(
    tiff_folder: os.PathLike,
    mask: gp.GeoDataFrame = "",
    overwrite: bool = False,
) -> os.PathLike:
    """Gets the path to the merged tiff, if not existent merges the tiffs in `tiff_folder`

    :param tiff_folder: The path to a folder containing tiffs
    :type tiff_folder: os.PathLike
    :param mask: The mask to cut the merged file, defaults to ""
    :type mask: gp.GeoDataFrame, optional
    :param overwrite: Whether to overwrite existing results, defaults to False
    :type overwrite: bool, optional
    :return: The path to the merged tiff file.
    :rtype: os.PathLike
    """
    logging.info("Getting merged and cut tiff files")
    logging.info("Setting paths")
    base_folder, folder_name = os.path.split(tiff_folder)
    merged_folder_path = os.path.join(base_folder, "merged_tiffs")
    merged_file_path = os.path.join(
        merged_folder_path, f"{folder_name}_merged.tif"
    )
    if not os.path.exists(merged_file_path) or overwrite:
        logging.info("No files found or overwrite=True")
        tiff_file_list = [
            rio.open(os.path.join(tiff_folder, tf), "r")
            for tf in tqdm(
                os.listdir(tiff_folder),
                desc="Reading Tiffs for Merge",
                leave=False,
            )
            if tf.endswith(".tif")
        ]
        if len(mask) > 0:
            logging.info("Merging with mask")
            bounds = tuple(np.squeeze(mask.bounds.values))
            riomerge(tiff_file_list, dst_path=merged_file_path, bounds=bounds)
        else:
            logging.info("Merging without mask")
            riomerge(tiff_file_list, dst_path=merged_file_path)
        for tf in tiff_file_list:
            tf.close()
    return merged_file_path


def extract_xyz_tiff(
    file_path: os.PathLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Extracts the data and coordinates from a tiff file for easier processing with numpy.

    :param file_path: The path to the tiff file.
    :type file_path: os.PathLike
    :return: A Tuple of `np.ndarrays` with X, Y, Z values and the coordinate system.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, str]
    """
    logging.info("Reading data from tiff file.")
    with rio.open(file_path, "r") as tile:
        logging.info("Loading tile data")
        zz = np.squeeze(tile.read())
        crs = tile.crs.to_string()
        shp = zz.shape

        logging.info("Reading coordinates")
        x = []
        for ii in range(shp[1]):
            x.append(tile.xy(0, ii)[0])
        y = []
        for ii in range(shp[0]):
            y.append(tile.xy(ii, 0)[1])
    xx, yy = np.meshgrid(x, y)
    return (xx, yy, zz, crs)


def get_thresholded_contours(
    tiff_file_path: os.PathLike,
    levels: list,
    threshold: float,
    overwrite: bool = False,
) -> gp.GeoDataFrame:
    """Gets contours around regions of specific ground motions (`levels`) that
    are larger than the `threshold` area (in m²). Data is stored as a
    shapefile for faster access

    :param tiff_file_path: The input tiff file to create contours.
    :type tiff_file_path: os.PathLike
    :param levels: The levels of the contours
    :type levels: list
    :param threshold: The minimum area for detection
    :type threshold: float
    :param overwrite: Whether to overwrite the existing results, defaults to False
    :type overwrite: bool, optional
    :return: A GeoDataFrame containing polygon shapes including some more info.
    :rtype: gp.GeoDataFrame
    """
    logging.info("Getting thresholded contours")

    logging.info("Setting up paths")
    base_path, fname = os.path.split(tiff_file_path)
    out_folder = os.path.join(base_path, "thresholded tiffs")
    os.makedirs(out_folder, exist_ok=True)
    fname = os.path.splitext(fname)[0]
    out_fname = fname.replace("clipped_tiffs", "thresh_contours")
    out_path = os.path.join(out_folder, out_fname + ".shp")

    if os.path.exists(out_path) and not overwrite:
        logging.info("Loading from file")
        gdf = gp.read_file(out_path)
    else:
        logging.info("Generating new results")
        xx, yy, zz, crs = extract_xyz_tiff(tiff_file_path)
        gdf = u4spatial.contour_shapes(xx, yy, zz, levels, threshold, crs)
        gdf.to_file(out_path)
    return gdf


def get_osm_as_shp(
    tags: dict,
    project: dict,
    query: str = "Hesse",
    shape_type: shapely.GeometryType = shapely.Point,
    overwrite: bool = False,
) -> gp.GeoDataFrame:
    """Loads point data from openstreetmap in the `query` region with the `tags`. Saves results in a shape file.

    :param tags: The tags for open street map (same as for the overpass API)
    :type tags: dict
    :param project: The loaded project file (for path management)
    :type project: dict
    :param query: The region where to get the data, defaults to "Hesse"
    :type query: str, optional
    :param shape_type: The type of the shape data, defaults to `shapely.Point`
    :type shape_type: shapely.GeometryType, optional
    :param overwrite: Whether to overwrite existing results, defaults to False
    :type overwrite: bool, optional
    :return: A GeoDataFrame with the data
    :rtype: gp.GeoDataFrame
    """
    logging.info("Preparing Query")
    fname = ""
    for kk in tags.keys():
        if len(fname) > 0:
            fname += "_"

        fname += f"{kk[:3]}"

        if isinstance(tags[kk], list):
            substr = ""
            for ii, ll in enumerate(tags[kk]):
                if ii == 0:
                    substr += "-"
                else:
                    substr += "+"
                substr += ll[:3]
            fname += substr
        else:
            fname += f"-{tags[kk][:3]}"
    fname += ".shp"

    file_path = os.path.join(project["paths"]["places_path"], fname)

    if os.path.exists(file_path) and not overwrite:
        logging.info("Loading from shape file")
        slc_gdf = gp.read_file(file_path)
    else:
        logging.info("Downloading data from OSM")
        gdf = osmnx.features_from_place(
            query=query,
            tags=tags,
        ).to_crs("32632")
        logging.info("Converting geometry")
        gdf = gdf[[isinstance(g, shape_type) for g in gdf.geometry]]
        slices = []
        for kk in tags.keys():
            if isinstance(tags[kk], list):
                sub_slice = [gdf[kk] == ll for ll in tags[kk]]
                slices.append(np.logical_or(*sub_slice))
            else:
                slices.append(gdf[kk] == tags[kk])
        if len(slices) > 1:
            slc_gdf = gdf[np.logical_and(*slices)]
        else:
            slc_gdf = gdf[slices[0]]
        if shape_type == shapely.Polygon:
            rows, cols = gdf.values.shape
            slc_gdf = gp.GeoDataFrame(
                geometry=slc_gdf.geometry, crs=slc_gdf.crs
            )

        logging.info("Saving to shape file.")
        slc_gdf.to_file(file_path)
    return slc_gdf


def fiona_load(shp_path: os.PathLike, fclasses: list = []) -> gp.GeoDataFrame:
    logging.info("Reading with fiona.")
    with fiona.open(shp_path, "r") as shapefile:
        if "fclass" in shapefile.keys() and fclasses:
            geometries = [
                shpgeo.shape(feature.geometry)
                for feature in tqdm(
                    shapefile,
                    desc="Reading selected shapes from file",
                    leave=False,
                )
                if feature.properties["fclass"] in fclasses
            ]
        else:
            geometries = [
                shpgeo.shape(feature.geometry)
                for feature in tqdm(
                    shapefile, desc="Reading all shapes from file", leave=False
                )
            ]
        crs = shapefile.crs
    return geometries, crs