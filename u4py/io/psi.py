"""
Functions to work with PSI data, e.g. loading and querying data in specific
areas.

Makes use of functions from `gpkg` and `sql`.
"""

import os
import pickle as pkl
from typing import Tuple

import geopandas as gp
import numpy as np
import shapely

import u4py.analysis.spatial as u4spatial
import u4py.io.files as u4files
import u4py.io.gpkg as u4gpkg


def get_osm_data(
    query: dict,
    source_fpath: os.PathLike,
    overwrite: bool = False,
    crs: str = "EPSG:32632",
) -> dict:
    """Gets the data from PSI points in a region defined by an OSM query and saves it into a pickle file for faster access.

    :param query: The OSM query
    :type query: dict
    :param source_fpath: Path to folder or file where the PSI data is found.
    :type source_fpath: os.PathLike
    :param overwrite:  Whether to overwrite the output pickle file, defaults to False
    :type overwrite: bool, optional
    :param crs: The CRS of the input shapes, defaults to "EPSG:32632"
    :type crs: str, optional
    :return: The data as a merged, single dictionary.
    :rtype: dict
    """
    _, pkl_fpath = u4files.set_data_file_paths(
        source_fpath, query["address"], "osm"
    )
    _, shp_fpath = u4files.set_point_file_paths(
        source_fpath, query["address"], "osm"
    )
    _, region_fpath = u4files.set_point_file_paths(
        source_fpath, query["address"], "osm_region"
    )

    if (
        not os.path.exists(pkl_fpath)
        or not os.path.exists(region_fpath)
        or overwrite
    ):
        data, region = u4gpkg.load_gpkg_data_osm(query, source_fpath, crs=crs)
        u4spatial.xy_data_to_gdf(data["x"], data["y"], crs=crs).to_file(
            shp_fpath
        )
        region.to_file(region_fpath)
        with open(pkl_fpath, "wb") as pkl_file:
            pkl.dump(data, pkl_file)
    else:
        with open(pkl_fpath, "rb") as pkl_file:
            data = pkl.load(pkl_file)
        region = gp.read_file(region_fpath)
    return data, region


def get_region_data(
    region: gp.GeoDataFrame,
    region_name: str,
    source_fpath: os.PathLike,
    overwrite: bool = False,
    crs: str = "EPSG:32632",
) -> dict:
    """Gets the data from PSI points in a region and saves them into a pickle file for faster access.

    :param region: A `GeoDataFrame` of the region, e.g. from a shape file
    :type region: gp.GeoDataFrame
    :param region_name: The name of the region.
    :type region_name: str
    :param source_fpath: Path to folder or file where the PSI data is found.
    :type source_fpath: os.PathLike
    :param overwrite:  Whether to overwrite the output pickle file, defaults to False
    :type overwrite: bool, optional
    :param crs: The CRS of the input shapes, defaults to "EPSG:32632"
    :type crs: str, optional
    :return: The data as a merged, single dictionary.
    :rtype: dict
    """

    _, pkl_fpath = u4files.set_data_file_paths(
        source_fpath, region_name, "region"
    )
    _, shp_fpath = u4files.set_point_file_paths(
        source_fpath, region_name, "region"
    )

    if not os.path.exists(pkl_fpath) or overwrite:
        data = u4gpkg.load_gpkg_data_region(
            region, source_fpath, table="vertikal"
        )
        u4spatial.xy_data_to_gdf(data["x"], data["y"], crs=crs).to_file(
            shp_fpath
        )
        with open(pkl_fpath, "wb") as pkl_file:
            pkl.dump(data, pkl_file)
    else:
        with open(pkl_fpath, "rb") as pkl_file:
            data = pkl.load(pkl_file)
    return data


def get_point_data(
    point: Tuple[float, float] | shapely.Point,
    radius: float,
    region_name: str,
    source_fpath: os.PathLike,
    overwrite: bool = False,
) -> dict:
    """Returns the points with data from `source_fpath` that are within
    `radius` of `point`. Creates a pkl file containing the data for faster loading.

    :param point: The point where to start.
    :type point: Tuple[float, float]
    :param radius: The search radius around point (in meters).
    :type radius: float
    :param region_name: A sensible name for the extraction point.
    :type region_name: str
    :param source_fpath: The file where to extract the data.
    :type source_fpath: os.PathLike
    :param overwrite: Whether to overwrite the shape file, defaults to False
    :type overwrite: bool, optional
    :return: The data as a merged, single dictionary.
    :rtype: dict
    """
    if isinstance(point, shapely.Point):
        point = point.xy
    _, pkl_fpath = u4files.set_data_file_paths(
        source_fpath, region_name, "point"
    )
    _, shp_fpath = u4files.set_point_file_paths(
        source_fpath, region_name, "point"
    )
    _, region_fpath = u4files.set_point_file_paths(
        source_fpath, region_name, "point_region"
    )

    if (
        not os.path.exists(pkl_fpath)
        or not os.path.exists(region_fpath)
        or overwrite
    ):
        data, region = u4gpkg.load_gpkg_data_point(
            point=point, radius=radius, gpkg_file_path=source_fpath
        )
        u4spatial.xy_data_to_gdf(data["x"], data["y"]).to_file(shp_fpath)
        region.to_file(region_fpath)
        with open(pkl_fpath, "wb") as pkl_file:
            pkl.dump(data, pkl_file)
    else:
        with open(pkl_fpath, "rb") as pkl_file:
            data = pkl.load(pkl_file)
        region = gp.read_file(region_fpath)
    return data, region


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
        closest = u4spatial.spatial_lookup(data, points)
        data = [data[c[1]] for c in closest]
        distance = np.mean([c[0] for c in closest])
        return data, distance
    else:
        return data, 0
