"""
Contains several functions to perform spatial operations, such as reprojecting
or spatial lookup of features.
"""

from __future__ import annotations

import logging
import os
import pickle as pkl
import re
from typing import Iterable, List, Tuple

import fiona
import geopandas as gp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import osmnx
import rasterio as rio
import rasterio.coords
import rasterio.warp as riowarp
import scipy.spatial as spspatial
import shapely
import shapely.geometry as shpgeo
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from pyproj import Transformer
from skimage import measure as skmeasure
from tqdm import tqdm

import u4py.utils.sql as u4sql


def reproject_raster(
    in_path: os.PathLike,
    out_path: os.PathLike,
    output_crs: str,
) -> os.PathLike:
    """Reprojects the raster file in `in_path` to the given crs.

    :param in_path: The path to the rasterfile.
    :type in_path: os.PathLike
    :param out_path: The new file path of the reprojected raster file.
    :type out_path: os.PathLike
    :param output_crs: The output CRS string.
    :type output_crs: str
    :return: The path to the output file.
    :rtype: os.PathLike
    """
    crs_out = {"init": output_crs}
    with rio.open(in_path) as src:
        transform, width, height = riowarp.calculate_default_transform(
            src.crs, crs_out, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()

        kwargs.update(
            {
                "crs": crs_out,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                riowarp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_out,
                    resampling=riowarp.Resampling.nearest,
                )
    return out_path


def get_cKDTree(file_path: os.PathLike) -> spspatial.cKDTree:
    """Loads all x and y coordinates from the given file or folder and returns a cKDTree for easy spatial lookup.

    :param file_path: The path to the folder or file.
    :type file_path: os.PathLike
    :return: The spatial lookup Tree object
    :rtype: spspatial.cKDTree
    """
    coords, source_index = _get_coords(file_path)

    return spspatial.cKDTree(coords)


def _get_coords(in_path: os.PathLike | list) -> list:
    """
    Loads all x and y coordinates from the given file or folder.

    Works with single hdf or pkl files as well as folders with many files.

    :param in_path: The path to the folder or file.
    :type in_path: PathLike | list
    :raises NotImplementedError: Error raised when method to generate cKDTree from this filetype is not implemented.
    :return: A list of coordinates to use for generating the cKDTree [(x1, y1), (x2, y2),...].
    :rtype: list
    """
    coords = None

    if isinstance(in_path, list):
        if isinstance(in_path[0], str):
            coords, source_index = _file_list_to_coords(in_path)
        elif isinstance(in_path[0], tuple):
            coords = _tuple_list_to_coords(in_path)
            source_index = np.arange(len(coords))

    elif isinstance(in_path, str):
        if in_path.endswith(".h5"):
            coords = _h5_to_coords(in_path)
            source_index = np.arange(len(coords))
        elif in_path.endswith(".pkl"):
            coords = _pkl_to_coords(in_path)
            source_index = np.arange(len(coords))
        elif os.path.isdir(in_path):
            coords, source_index = _file_list_to_coords(in_path)

    if coords is None:
        raise NotImplementedError(
            "Conversion of this type to lookup table not implemented!"
        )
    return coords, source_index


def _file_list_to_coords(
    in_path: os.PathLike | list, ending: str = ".h5"
) -> list:
    """Converts a file list of files with x and y coordinates in their names
    to a list of coordinates.

    :param in_path: The folder containing the files.
    :type in_path: os.PathLike
    :return: List of coordinates [(x1, y1), (x2, y2),...].
    :rtype: list
    """
    if isinstance(in_path, str) and os.path.isdir(in_path):
        coords = [
            shapely.Point(
                int(f[f.find("_x") + 2 : f.find("_y")]),
                int(f[f.find("_y") + 2 : f.find(ending)]),
            )
            for f in os.listdir(in_path)
            if f.endswith(ending)
        ]
        source_index = np.arange(len(coords))

    elif isinstance(in_path, list):
        coords = []
        source_index = []
        for ii, fpath in enumerate(in_path):
            if fpath.endswith(".tif"):
                coords.extend(_bounds_to_coords(fpath))
                source_index.extend([ii] * 4)
            else:
                fname, _ = os.path.splitext(os.path.split(fpath)[-1])
                ind = [m.start() for m in re.finditer("_", fname)]
                coords.append(
                    shapely.Point(
                        int(fname[ind[1] + 1 : ind[2]]) * 1000,
                        int(fname[ind[2] + 1 : ind[3]]) * 1000,
                    )
                )
                source_index.append(ii)
    else:
        NotImplementedError(f"Conversion of {in_path} not supported.")

    return coords, source_index


def _bounds_to_coords(fpath: os.PathLike, tilesize=1000) -> list:
    """Converts the bounds of a GeoTiff to four `shapely.Points` representing the corners of the Box.

    :param fpath: The path to a GeoTiff
    :type fpath: os.PathLike
    :return: The corners as a list of `shapely.Points`
    :rtype: list
    """
    fname, _ = os.path.splitext(os.path.split(fpath)[-1])
    ind = [m.start() for m in re.finditer("_", fname)]
    left = int(fname[ind[1] + 1 : ind[2]]) * 1000
    bottom = int(fname[ind[2] + 1 : ind[3]]) * 1000
    right = left + tilesize
    top = bottom + tilesize
    coords = [
        shapely.Point(left, bottom),
        shapely.Point(left, top),
        shapely.Point(right, bottom),
        shapely.Point(right, top),
    ]
    return coords


def _h5_to_coords(h5file_path: os.PathLike) -> list:
    """Converts a h5 file containing x and y to list of coordinates.

    :param h5file_path: The path to the h5 file.
    :type h5file_path: os.PathLike
    :return: List of coordinates [(x1, y1), (x2, y2),...].
    :rtype: list
    """
    with h5py.File(h5file_path, "r") as h5file:
        coords = np.array(
            [(x, y) for x, y in zip(h5file["x"][()], h5file["y"][()])]
        )
    return coords


def _tuple_list_to_coords(tuple_list: list) -> list:
    """Converts a list of tuples from a loaded pkl file to list of coordinates.

    :param tuple_list: List of xy tuples
    :type tuple_list: list
    :return: List of coordinates [(x1, y1), (x2, y2),...].
    :rtype: list
    """
    coords = np.array([(d[0], d[1]) for d in tuple_list])
    return coords


def _pkl_to_coords(pkl_path: os.PathLike) -> list:
    """Converts contents of a pickled inversion results file to list of
    coordinates.

    :param pkl_path: The path to the pkl file.
    :type pkl_path: os.PathLike
    :return: List of coordinates [(x1, y1), (x2, y2),...].
    :rtype: list
    """
    with open(pkl_path, "rb") as pklfile:
        data = pkl.load(pklfile)[0]
    coords = _tuple_list_to_coords(data)
    return coords


def spatial_lookup(
    input_feature: os.PathLike | list, points: gp.GeoDataFrame, n: int = 1
) -> List[Tuple[float, int]]:
    """Does a spatial lookup for the nearest point to all points in `points`.
    The data at `input_feature` has to have a valid format for creating a  lookup Tree.

    :param input_feature: A variable that contains some sort of x and y table which can be converted to a lookup table with :func:`get_cKDTree`
    :type input_feature: os.PathLike | list
    :param points: Point features to query.
    :type points: gp.GeoDataFrame
    :param n: The number of nearest neighbors, defaults to 1
    :type n: int, optional
    :return: A list of `n` nearest neighbors for each point in the form of (distance, index)
    :rtype: List[Tuple[float, int]]
    """

    lut = get_cKDTree(input_feature)
    points = points.values
    closest = [lut.query((p[1].x, p[1].y), n) for p in points]
    return closest


def select_points_osm(
    osm_query: dict, psi_file_path: os.PathLike, crs: str = "EPSG:32632"
) -> gp.GeoDataFrame:
    """Selects PSI measurements from the specified file or folder from the specified file or files in `psi_file_path` and crops them by the rectangles found in the given osm query.

    This function is part of a two step method. First points are extracted and then later the data is taken using `u4files.load_data_from_points()`.

    :param osm_query: A properly formatted osm query in dictionary form. (see https://osmnx.readthedocs.io/en/stable/ for more)
    :type osm_query: dict
    :param psi_file_path: The files or folder where the psi data is stored (e.g., a folder containing h5 files)
    :type psi_file_path: os.PathLike
    :return: A GeoDataFrame containing all points within the openstreetmap geometry.
    :rtype: gp.GeoDataFrame
    """

    osm_data = get_osm_region(osm_query, crs)

    coords, source_index = _get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": source_index,
        },
        crs=crs,
    )
    return points.clip(osm_data)


def get_osm_region(
    osm_query: dict, crs: str = "EPSG:32632"
) -> gp.GeoDataFrame:
    """Loads a region from a OSM query.

    :param osm_query: A properly formatted osm query in dictionary form. (see https://osmnx.readthedocs.io/en/stable/ for more)
    :type osm_query: dict
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The region as a geodataframe
    :rtype: gp.GeoDataFrame
    """
    logging.info("Loading region from OSM")
    osm_data = osmnx.features_from_address(
        osm_query["address"], tags=osm_query["tags"]
    ).to_crs(crs)
    osm_data = gp.GeoDataFrame(geometry=[osm_data.unary_union], crs=crs)
    return osm_data


def select_points_region(
    region: gp.GeoDataFrame | gp.GeoSeries,
    psi_file_path: os.PathLike,
    crs: str = "",
) -> gp.GeoDataFrame:
    """Selects PSI measurements from the specified file or folder and crops them by the rectangles found in the given region.

    This function is part of a two step method. First points are extracted and then later the data is taken using `u4files.load_data_from_points()`.

    :param region: The region to crop the data (e.g, a loaded shapefile)
    :type region: gp.GeoDataFrame | gp.GeoSeries
    :param psi_file_path: The file or folder to select from
    :type psi_file_path: os.PathLike
    :param crs: The CRS of the input region, defaults to ""
    :type crs: str, optional
    :return: The points from `psi_file_path` cropped to the `region`.
    :rtype: gp.GeoDataFrame
    """

    coords, source_index = _get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": source_index,
        },
        crs="EPSG:32632",
    )
    if region.crs != points.crs:
        region = region.to_crs(points.crs)
    elif crs:
        region = gp.GeoDataFrame(
            geometry=gp.GeoSeries(region.geometry), crs=crs
        )
    return points.clip(region)


def select_points_point(
    point: list | Tuple | gp.GeoDataFrame | shapely.Point,
    radius: float,
    psi_file_path: os.PathLike,
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> gp.GeoDataFrame | List[gp.GeoDataFrame]:
    """Selects PSI measurements in a `radius` around the specified `point` from the files or folder.

    This function is part of a two step method. First points are extracted and then later the data is taken using `u4files.load_data_from_points()`.

    :param point: The center point of the query.
    :type point: list
    :param radius: The radius to calculate the buffer
    :type radius: float
    :param psi_file_path: The file or folder to select from
    :type psi_file_path: os.PathLike
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The points from `psi_file_path` in a `radius` round `point`.
    :rtype: gp.GeoDataFrame
    """

    # Get region for clipping
    region = region_around_point(point, radius, crs=crs)
    # Calculate Points
    coords, source_index = _get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": source_index,
        },
        crs=crs,
    )

    if split_points:
        return [points.clip(reg) for reg in region]
    else:
        return points.clip(region)


def load_gpkg_data_point(
    point: list | Tuple | gp.GeoDataFrame | shapely.Point,
    radius: float,
    gpkg_file_path: os.PathLike,
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> gp.GeoDataFrame | List[gp.GeoDataFrame]:
    """Selects PSI measurements in a `radius` around the specified `point` from the given **GPKG** File.

    This function loads the data directly using sql, no second step is required.

    :param point: The center point of the query.
    :type point: list
    :param radius: The radius to calculate the buffer
    :type radius: float
    :param gpkg_file_path: The file or folder to select from
    :type gpkg_file_path: os.PathLike
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The points from `psi_file_path` in a `radius` round `point`.
    :rtype: gp.GeoDataFrame
    """

    region = region_around_point(point, radius, crs=crs)

    data = u4sql.table_to_dict(gpkg_file_path, "vertikal", region.bounds)
    clipped_data = clip_data_points(
        data, region, split_points=split_points, crs=crs
    )
    return clipped_data


def load_gpkg_data_osm(
    osm_query: dict,
    gpkg_file_path: os.PathLike,
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> gp.GeoDataFrame | List[gp.GeoDataFrame]:
    """Selects PSI measurements in a region defined by an OSM query from the given **GPKG** File.

    This function loads the data directly using sql, no second step is required.

    :param osm_query: A properly formatted osm query in dictionary form. (see https://osmnx.readthedocs.io/en/stable/ for more)
    :type osm_query: dict
    :param gpkg_file_path: The file or folder to select from
    :type gpkg_file_path: os.PathLike
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The points from `psi_file_path` in a `radius` round `point`.
    :rtype: gp.GeoDataFrame
    """

    region = get_osm_region(osm_query, crs=crs)

    data = u4sql.table_to_dict(
        gpkg_file_path, "vertikal", bounds=region.bounds
    )
    clipped_data = clip_data_points(
        data, region, split_points=split_points, crs=crs
    )
    return clipped_data


def load_gpkg_data_region(
    region: gp.GeoDataFrame | gp.GeoSeries,
    gpkg_file_path: os.PathLike,
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> gp.GeoDataFrame | List[gp.GeoDataFrame]:
    """Selects PSI measurements in a `radius` around the specified `point` from the given **GPKG** File.

    This function loads the data directly using sql, no second step is required.

    :param osm_query: A properly formatted osm query in dictionary form. (see https://osmnx.readthedocs.io/en/stable/ for more)
    :type osm_query: dict
    :param gpkg_file_path: The file or folder to select from
    :type gpkg_file_path: os.PathLike
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The points from `psi_file_path` in a `radius` round `point`.
    :rtype: gp.GeoDataFrame
    """
    if isinstance(region, gp.GeoDataFrame):
        if region.crs != crs:
            region = region.to_crs(crs)
    elif isinstance(region, (gp.pd.Series, gp.GeoSeries)):
        UserWarning(
            "Region for selection is a GeoSeries, check input CRS manually!"
        )
        region = gp.GeoDataFrame(geometry=[region.geometry], crs=crs)

    data = u4sql.table_to_dict(
        gpkg_file_path, "vertikal", bounds=region.bounds
    )
    clipped_data = clip_data_points(
        data, region, split_points=split_points, crs=crs
    )
    return clipped_data


def clip_data_points(
    data: dict,
    region: gp.GeoDataFrame,
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> dict:
    """Clips the input data dictionary by the region and returns a GeoDataBase

    :param data: The data dictionary loaded by `u4sql.table_to_dict`.
    :type data: dict
    :param region: The region to use for clipping
    :type region: gp.GeoDataFrame
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :type split_points: bool, optional
    :return: The data generated from the input data and clipped by region.
    :rtype: dict
    """
    points = xy_data_to_gdf(
        data["x"], data["y"], data["timeseries"], z=data["z"], crs=crs
    )
    points = points.assign(ps_id=data["ps_id"])

    if split_points:
        return [
            clipped_data_to_dict(points.clip(reg), data["time"])
            for reg in region
        ]
    else:
        return clipped_data_to_dict(points.clip(region), data["time"])


def clipped_data_to_dict(
    clipped_data: gp.GeoDataFrame, time: np.ndarray
) -> dict:
    """Converts the clipped GeoDataFrame into a dictionary for inversion

    :param clipped_data: The clipped dataframe
    :type clipped_data: gp.GeoDataFrame
    :param time: A time array
    :type time: np.ndarray
    :return: Dictionary ready for use with inversion.
    :rtype: dict
    """
    loaded_data = {
        "num_points": len(clipped_data),
        "ps_id": clipped_data.ps_id.to_numpy(),
        "x": clipped_data.geometry.x.to_numpy(),
        "xmid": np.mean(clipped_data.geometry.x.to_numpy()),
        "y": clipped_data.geometry.y.to_numpy(),
        "ymid": np.mean(clipped_data.geometry.y.to_numpy()),
        "z": clipped_data.geometry.z.to_numpy(),
        "zmid": np.mean(clipped_data.geometry.z.to_numpy()),
        "timeseries": np.vstack(clipped_data.data.to_numpy()),
        "time": time,
    }
    return loaded_data


def region_around_point(
    point: list | Tuple | gp.GeoDataFrame | shapely.Point,
    radius: float,
    crs: str = "EPSG:32632",
) -> gp.GeoDataFrame:
    """Calculates the region around a point

    :param point: The center point of the query.
    :type point: list
    :param radius: The radius to calculate the buffer
    :type radius: float
    "EPSG:32632".
    :type crs: str, optional
    :return: The region as a geodataframe
    :rtype: gp.GeoDataFrame
    """
    if isinstance(point, (list, tuple)):
        region = gp.GeoDataFrame(
            {"geometry": [shapely.Point(point).buffer(radius)]},
            crs=crs,
        )
    elif isinstance(point, (gp.GeoDataFrame, gp.GeoSeries)):
        if point.crs != crs:
            point = point.to_crs(crs)
        region = point.buffer(radius)
    elif isinstance(point, shapely.Point):
        region = point.buffer(radius)
    return region


def bounds_to_polygon(bounds: rasterio.coords.BoundingBox) -> shapely.Polygon:
    """Creates a `shapely.Polygon` from the `BoundingBox` of loaded raster.

    :param bounds: The `BoundingBox` of a raster.
    :type bounds: rasterio.coords.BoundingBox
    :return: The polygon.
    :rtype: shapely.Polygon
    """
    poly = shapely.Polygon(
        shell=(
            (bounds.left, bounds.bottom),
            (bounds.left, bounds.top),
            (bounds.right, bounds.top),
            (bounds.right, bounds.bottom),
            (bounds.left, bounds.bottom),
        )
    )
    return poly


def river_coordinates_from_osm(
    station_name: str, waterway_name: str
) -> gp.GeoSeries:
    """Gets the river line feature from openstreetmap.

    :param station_name: The name of the station (city).
    :type station_name: str
    :param waterway_name: The name of the waterway.
    :type waterway_name: str
    :return: A geoseries, usually several line features, of the river cropped to the administrative boundaries of the station.
    :rtype: gp.GeoSeries
    """
    waterway_osm = osmnx.features_from_place(
        station_name + " Hesse",
        tags={"waterway": ["stream", "river"]},
    )
    if not waterway_osm.empty:
        waterway_geoseries = get_named_geometries(waterway_osm, waterway_name)
        if not waterway_geoseries.empty:
            region_osm = osmnx.features_from_place(
                station_name + " Hesse", tags={"boundary": "administrative"}
            )
            if not region_osm.empty:
                region_geoseries = get_named_geometries(
                    region_osm, station_name
                )
                buffered_region = (
                    region_geoseries.to_crs("EPSG:32632")
                    .buffer(25)
                    .to_crs(region_geoseries.crs)
                )
                return gp.clip(waterway_geoseries, buffered_region)
            else:
                print(
                    f"Geometry not clipped. No osm administrative boundary found for {station_name}"
                )
                return waterway_geoseries
        else:
            print(f"No geometry found for {waterway_name} in {station_name}")
    else:
        print(f"No OSM Data for {waterway_name} in {station_name}")


def get_named_geometries(
    osm_data: gp.GeoDataFrame, feature_name: str
) -> gp.GeoSeries:
    """Gets geometries with the given feature name from the osm data.

    :param osm_data: The dataset as loaded from osm
    :type osm_data: gp.GeoDataFrame
    :param feature_name: The exact name of the feature to be extracted.
    :type feature_name: str
    :return: A geoseries with the geometries.
    :rtype: gp.GeoSeries
    """
    geometries = []
    for ii, name in enumerate(osm_data.name):
        if name == feature_name:
            geometries.append(osm_data.geometry[ii])
    if geometries:
        return gp.GeoSeries(geometries, crs=osm_data.crs)


def catalog_to_gdf(
    data: dict, mask: gp.GeoDataFrame = None
) -> gp.GeoDataFrame:
    """Converts an earthquake catalogue to a geodataframe.

    :param data: The data dictionary
    :type data: dict
    :param mask: An optional masking GeoDataFrame, defaults to None
    :type mask: gp.GeoDataFrame, optional
    :return: The (masked) GeoDataFrame
    :rtype: gp.GeoDataFrame
    """
    data["geometry"] = [
        shapely.Point(x, y) for x, y in zip(data["LAENGE"], data["BREITE"])
    ]

    gdf = gp.GeoDataFrame(data, crs="EPSG:4326").to_crs("EPSG:23032")
    if isinstance(mask, gp.GeoDataFrame):
        if mask.crs != "EPSG:23032":
            mask.to_crs("EPSG:23032")
        return gp.clip(gdf, mask)
    else:
        return gdf


def calculate_buffer(
    gdf: gp.GeoDataFrame, buffer_size: float, crs: str = "EPSG:32632"
) -> gp.GeoDataFrame:
    """Wrapper for buffer function that automatically switches to a projected CRS

    :param gdf: The input GeoDataFrame
    :type gdf: gp.GeoDataFrame
    :param buffer_size: The buffer size in metres
    :type buffer_size: float
    :return: The buffered GeoDataFrame
    :rtype: gp.GeoDataFrame
    """
    logging.info("Calculating buffers")
    old_crs = gdf.crs
    if old_crs != crs:
        gdf = gdf.to_crs(crs)
    gdf_buf = gdf.buffer(buffer_size)
    gdf_buf = gdf_buf.to_crs(old_crs)
    return gp.GeoDataFrame(geometry=gdf_buf, crs=old_crs)


def buffer_and_merge(shp_data: dict, shp_cfg: dict) -> gp.GeoDataFrame:
    """Creates buffers for all input shapes and merges them to a large `GeoDataFrame`

    :param shp_data: The dictionary with shape data
    :type shp_data: dict
    :param shp_cfg: The configuration for the shape data (include buffer sizes)
    :type shp_cfg: dict
    :return: The merged and buffered dataset.
    :rtype: gp.GeoDataFrame
    """

    buff_gdf = gp.pd.concat(
        [
            calculate_buffer(shp_data[kk], shp_cfg["buffer_dist"][kk])
            for kk in tqdm(shp_data.keys(), desc="Buffering")
        ]
    )
    return buff_gdf


def contour_shapes(
    coords: dict,
    zz: np.ndarray,
    levels: list,
    min_area: float,
    crs: str,
    delete_outside: bool = True,
) -> gp.GeoDataFrame:
    """Generates a geodataframe of `Polygons` from the `xx, yy, zz` dataset using the contour `levels`.

    :param coords: A dictionary containing lower left corner and dx, dy
    :type coords: dict
    :param zz: The array with displacements
    :type zz: np.ndarray
    :param levels: The levels to use for the contour plot
    :type levels: list
    :param min_area: The minimum area for a contour surface to be taken into account.
    :type min_area: float
    :param min_area: The coordinate system of the Polygons.
    :type min_area: str
    :param delete_outside: Whether to set values above the highest level to nan (and below the lowest), defaults to True
    :type delete_outside: bool, optional
    :return: A list of Polygons together with the levels they have been generated from.
    :rtype: gp.GeoDataFrame
    """
    logging.info("Generating Contour Shapes")

    if delete_outside:
        logging.debug("Deleting values outside level range")
        zz[np.abs(zz) > levels[-1]] = np.nan
        zz[np.abs(zz) < levels[0]] = np.nan

    logging.debug("Making contour set")
    contours = [
        skmeasure.find_contours(np.rot90(zz, 3), level=level)
        for level in tqdm(levels, desc="Getting Contours")
    ]

    logging.debug("Separating contour set")
    polygons = []  # The polygon shapes
    polygon_levels = []  # The levels of the polygon
    color_levels = []  # The midpoint of the levels for easy plotting
    for ii, segs in enumerate(contours):
        if len(segs) > 0:
            for p in segs:
                if len(p) > 3:
                    p_rescaled = [
                        (
                            coords["x"] + c[0] * coords["dx"],
                            coords["y"] - c[1] * coords["dy"],
                        )
                        for c in p
                    ]
                    pgon = shpgeo.Polygon(p_rescaled)
                    if pgon.area > min_area:
                        polygons.append(pgon)
                        polygon_levels.append(f"{levels[ii]} - {levels[ii+1]}")
                        color_levels.append(
                            np.mean([levels[ii], levels[ii + 1]])
                        )

    logging.debug("Generating GeoDataFrame")
    gdf = gp.GeoDataFrame(
        data={"contour": polygon_levels, "level": color_levels},
        geometry=polygons,
        crs=crs,
    )
    return gdf


def plus_minus_levels(half_sided: list) -> list:
    """Generates a mirrored levels for the half-sided level list, e.g. `[2,3,4]` -> `[-4,-3,-2,2,3,4]`

    :param half_sided: A list of levels to be mirrored
    :type half_sided: list
    :return: The mirrored levels
    :rtype: list
    """

    half_sided.extend([-1 * ll for ll in half_sided])
    half_sided.sort()
    return half_sided


def xy_data_to_gdf(
    x: np.ndarray | list,
    y: np.ndarray | list,
    data: np.ndarray | list = [],
    crs: str = "EPSG:32632",
    z: np.ndarray | list = [],
) -> gp.GeoDataFrame:
    """Converts the three input iterables to a GeoDataFrame of Points with the
    data as data column.

    :param x: The x coordinates.
    :type x: np.ndarray | list
    :param y: The y coordinates.
    :type y: np.ndarray | list
    :param data: The values, defaults to [].
    :type data: np.ndarray | list
    :param crs: The coordinate system, defaults to "EPSG:32632".
    :type crs: str
    :return: The data as geodataframe.
    :rtype: gp.GeoDataFrame
    """
    if len(z) > 0:
        geometry = [shapely.Point(xx, yy, zz) for xx, yy, zz in zip(x, y, z)]
    else:
        geometry = [shapely.Point(xx, yy) for xx, yy in zip(x, y)]

    gdf = gp.GeoDataFrame(geometry=geometry, crs=crs)
    if len(data) > 0:
        gdf = gdf.assign(data=[*data])
    return gdf


def hotspots_hexbin(
    vals: np.ndarray,
    coords: dict,
    thresh: float | Iterable,
    nbins: int = 3,
    min_count: int = 3,
) -> Tuple[Figure, Axes, PolyCollection]:
    """Uses matplotlibs hexbin to create a list of hotspots.

    These are areas in the data that are above the threshold and cover at least `nbins` x `(dx, dy)`.

    :param vals: A numpy array with the values to bin.
    :type vals: np.ndarray
    :param coords: A dictionary with minimum x, y and dx, dy.
    :type coords: dict
    :param thresh: Either a single threshold or an iterable with two thresholds. If it is iterable values below the lower threshold and above the higher threshold are detected as hotspots.
    :type thresh: float | Iterable
    :param nbins: The minimum area in n x (dx, dy) that a hotspot has to cover for detection, defaults to 3
    :type nbins: int
    :param min_count: The minimum number of psi in a bin to be detected, defaults to 3
    :return: The figure, axes and hexbin collection for further analysis.
    :rtype: Tuple[Figure, Axes, PolyCollection]
    """
    logging.info("Running HexagonalBinning")
    # Create grid for binning
    r, c = vals.shape
    x = np.arange(coords["x"], coords["x"] + (c * coords["dx"]), coords["dx"])
    y = np.arange(coords["y"], coords["y"] + (r * coords["dy"]), coords["dy"])
    xx, yy = np.meshgrid(x, y)

    # Thresholding
    if isinstance(thresh, Iterable):
        vals[(vals < np.max(thresh)) & (vals > np.min(thresh))] = np.nan
    else:
        vals[vals < thresh] = np.nan
    xx_real = xx[np.isfinite(vals)]
    yy_real = yy[np.isfinite(vals)]

    # Binning
    fig, ax = plt.subplots(figsize=(6, 10), dpi=150)
    h = ax.hexbin(
        xx_real,
        yy_real,
        gridsize=(int(c / nbins), int(r / nbins)),
        mincnt=min_count,
        alpha=0.75,
    )

    return (fig, ax, h)


def hotspots_GroundMotionAnalyzer(
    data: dict, cellsize: int, min_mean: float, max_var: float
) -> Tuple[Figure, Axes, PolyCollection]:
    logging.info("Running GroundMotionAnalyzer")

    # Slice data with minimum velocity and maximum variance
    slc = (np.abs(data["mean_vel"]) >= min_mean) & (
        data["var_mean_vel"] < max_var
    )
    for k in data.keys():
        data[k] = data[k][slc]

    # Create a histogram counting the values in a regular grid.
    x_edges = np.arange(
        np.min(data["x"]), np.max(data["x"]) + cellsize, cellsize
    )
    y_edges = np.arange(
        np.min(data["y"]), np.max(data["y"]) + cellsize, cellsize
    )
    fig, ax = plt.subplots()
    hist_grid, _, _, _ = ax.hist2d(
        data["x"], data["y"], bins=(x_edges, y_edges), cmin=1, alpha=0.75
    )
    plt.close(fig)

    # Create gridded numpy array and do a blockwise reduction using sums
    grid = scattered_to_gridded(data["x"], data["y"], np.abs(data["mean_vel"]))
    red_grid = skmeasure.block_reduce(
        grid, block_size=int(cellsize / 50), func=np.nansum
    )
    red_grid[red_grid == 0] = np.nan
    gma_grid = np.log10(hist_grid * red_grid)

    return gma_grid


def scattered_to_gridded(
    x: Iterable, y: Iterable, values: Iterable, dx: float = 50, dy: float = 50
) -> np.ndarray:
    """Converts x, y, z values to a gridded numpy array.

    :param x: 1D array of x coordinates
    :type x: Iterable
    :param y: 1D array of y coordinates
    :type y: Iterable
    :param values: 1D array of corresponding z values
    :type values: Iterable
    :return: 2D numpy array representing the gridded data
    :rtype: np.ndarray
    """
    logging.info("Converting scattered data to grid.")
    xr = np.arange(np.min(x), np.max(x) + dx, dx)
    yr = np.arange(np.min(y), np.max(y) + dy, dy)
    xi = np.searchsorted(xr, x, side="left")
    yi = np.searchsorted(yr, y, side="left")
    grid = np.ones((len(xr), len(yr))) * np.nan
    grid[xi, yi] = values

    return grid


def subdivide_polygon(
    roi: gp.GeoDataFrame,
    overlap: int = 250,
):
    if isinstance(roi, gp.GeoDataFrame):
        bounds = roi.bounds.to_numpy()[0]
    elif isinstance(roi, shapely.Polygon):
        bounds = roi.bounds
    minx = bounds[0]
    miny = bounds[1]
    maxx = bounds[2]
    maxy = bounds[3]
    lenx = (maxx - minx) / 2
    leny = (maxy - miny) / 2
    if lenx > leny:  # split along x axis
        new_poly1 = shapely.Polygon(
            [
                (minx, miny),
                (minx + lenx + overlap, miny),
                (minx + lenx + overlap, maxy),
                (minx, maxy),
                (minx, miny),
            ]
        )
        new_poly2 = shapely.Polygon(
            [
                (minx + lenx, miny),
                (minx + 2 * lenx + overlap, miny),
                (minx + 2 * lenx + overlap, maxy),
                (minx + lenx, maxy),
                (minx + lenx, miny),
            ]
        )
    else:  # split along y axis
        new_poly1 = shapely.Polygon(
            [
                (minx, miny),
                (maxx, miny),
                (maxx, miny + leny + overlap),
                (minx, miny + leny + overlap),
                (minx, miny),
            ]
        )
        new_poly2 = shapely.Polygon(
            [
                (minx, miny + leny),
                (maxx, miny + leny),
                (maxx, miny + 2 * leny + overlap),
                (minx, miny + 2 * leny + overlap),
                (minx, miny + leny),
            ]
        )

    return (new_poly1, new_poly2)


def buffer_features(
    feature: fiona.Feature,
    distance: float,
    transformer: Transformer,
    fshapes: dict,
) -> list[shapely.Polygon]:
    """Buffers the given feature depending on its feature type. If it is not

    :param feature: The feature for buffering
    :type feature: fiona.Feature
    :param distance: The buffer distance
    :type distance: float
    :param transformer: A transformer that maps the feature coordinates to the desired output CRS. If is None, then keeps the input CRS.
    :type transformer: Transformer
    :param fshapes: Dictionary mapping the Fiona shape types to shapely geometry.
    :type fshapes: dict
    :return: A list of polygons representing the buffers.
    :rtype: list[shapely.Polygon]
    """
    coords, isMulti = get_coords_fiona(feature)
    if not isMulti:
        if transformer:
            geometry = [
                buffer_geometry(
                    transform_coords(coords, transformer),
                    fshapes,
                    feature.geometry.type,
                    distance,
                )
            ]
        else:
            geometry = [
                buffer_geometry(
                    coords, fshapes, feature.geometry.type, distance
                )
            ]
    else:
        if transformer:
            geometry = [
                buffer_geometry(
                    transform_coords(np.array(coord[0]), transformer),
                    fshapes,
                    feature.geometry.type,
                    distance,
                )
                for coord in coords
            ]
        else:
            geometry = [
                buffer_geometry(
                    np.array(coord[0]),
                    fshapes,
                    feature.geometry.type,
                    distance,
                )
                for coord in coords
            ]
    return geometry


def transform_coords(
    coords: np.ndarray, transformer: Transformer
) -> np.ndarray:
    """Transforms the coordinates in `coords` using `transformer`. The
    transformer is preinitialized with input and output CRS. This is
    presumably faster than creating a geopandas geometry and calling `to_crs`.
    Actually, this is similar to how it is implemented in geopandas.

    :param coords: The coordinates a two dimensional numpy array.
    :type coords: np.ndarray
    :param transformer: A transformer object converting from one to another CRS.
    :type transformer: Transformer
    :return: An array with transformed coordinates.
    :rtype: np.ndarray

    *Does not support 3D geometries*
    """
    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    return np.array([(xx, yy) for xx, yy in zip(x, y)])


def buffer_geometry(
    coords: np.ndarray, fshapes: dict, ftype: str, distance: float
) -> shapely.Polygon:
    """Buffers the input geometry according to the feature class by distance.

    :param coords: An array of coordinates.
    :type coords: np.ndarray
    :param fshapes: A dictionary mapping Fiona feature types to shapely geometries types, e.g. `{"Polygon": shapely.Polygon}`.
    :type fshapes: dict
    :param ftype: The feature type.
    :type ftype: str
    :param distance: The distance to use for buffering.
    :type distance: float
    :return: The buffered shape.
    :rtype: shapely.Polygon
    """
    return fshapes[ftype](coords).buffer(distance)


def get_coords_fiona(feature):
    isMulti = False
    if feature.geometry.type != "MultiPolygon":
        coords = np.array(feature.geometry.coordinates[0])
    elif feature.geometry.type == "MultiPolygon":
        isMulti = True
        coords = feature.geometry.coordinates
    return coords, isMulti


def calculate_bounds(features: list[shapely.Geometry]) -> dict:
    """Calculates the boundaries of all geometries in the list. Returns a
    dictionary with the keys "min_x", "max_x", "min_y", "max_y".

    :param features: A list of geometries to determine the boundaries.
    :type features: list[shapely.Geometry]
    :return: The boundaries as a dictionary containing lists of the boundaries.
    :rtype: dict
    """
    bounds = {
        "min_x": [],
        "max_x": [],
        "min_y": [],
        "max_y": [],
    }
    if isinstance(features[0], shapely.Point):
        for feat in tqdm(features, "Getting bounds"):
            bounds["min_x"].append(feat.x)
            bounds["max_x"].append(feat.x)
            bounds["min_y"].append(feat.y)
            bounds["max_y"].append(feat.y)

    if isinstance(features[0], shapely.LineString):
        raise NotImplementedError
    if isinstance(features[0], shapely.MultiLineString):
        raise NotImplementedError
    if isinstance(features[0], shapely.Polygon):
        raise NotImplementedError
    if isinstance(features[0], shapely.MultiPolygon):
        raise NotImplementedError

    return bounds
