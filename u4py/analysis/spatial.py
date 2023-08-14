"""
Contains several functions to perform spatial operations, such as reprojecting
or spatial lookup of features.
"""

from __future__ import annotations

import os
import pickle as pkl
import re
from typing import List, Tuple

import geopandas as gp
import h5py
import numpy as np
import osmnx
import rasterio as rio
import rasterio.coords
import rasterio.warp as riowarp
import scipy.spatial as spspatial
import shapely
from tqdm import tqdm


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


def _spatial_lookup(
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


def _select_points_osm(
    osm_query: dict, psi_file_path: os.PathLike
) -> gp.GeoDataFrame:
    """Selects PSI measurements from the specified file or folder from the specified file or files in `psi_file_path` and crops them by the rectangles found in the given osm query.

    :param osm_query: A properly formatted osm query in dictionary form. (see https://osmnx.readthedocs.io/en/stable/ for more)
    :type osm_query: dict
    :param psi_file_path: The files or folder where the psi data is stored (e.g., a folder containing h5 files)
    :type psi_file_path: os.PathLike
    :return: A GeoDataFrame containing all points within the openstreetmap geometry.
    :rtype: gp.GeoDataFrame
    """

    osm_data = osmnx.features_from_address(
        osm_query["address"], tags=osm_query["tags"]
    ).to_crs("EPSG:32632")

    coords, source_index = _get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": source_index,
        },
        crs="EPSG:32632",
    )
    return points.clip(osm_data)


def _select_points_region(
    region: gp.GeoDataFrame | gp.GeoSeries,
    psi_file_path: os.PathLike,
    crs: str = "",
) -> gp.GeoDataFrame:
    """Selects PSI measurements from the specified file or folder and crops them by the rectangles found in the given region.

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


def _select_points_point(
    point: list | gp.GeoDataFrame,
    radius: float,
    psi_file_path: os.PathLike,
    split_points: bool = False,
) -> gp.GeoDataFrame | List[gp.GeoDataFrame]:
    """Selects PSI measurements in a `radius` around the specified `point` from the files or folder.

    :param point: The center point of the query.
    :type point: list
    :param radius: The radius to calculate the buffer
    :type radius: float
    :param psi_file_path: The file or folder to select from
    :type psi_file_path: os.PathLike
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :return: The points from `psi_file_path` in a `radius` round `point`.
    :rtype: gp.GeoDataFrame
    """
    if isinstance(point, list):
        region = gp.GeoDataFrame(
            {"geometry": [shapely.Point(point).buffer(radius)]},
            crs="EPSG:32632",
        )
    elif isinstance(point, gp.GeoDataFrame) or isinstance(point, gp.GeoSeries):
        if point.crs != "EPSG:32632":
            point = point.to_crs("EPSG:32632")
        region = point.buffer(radius)
    coords, source_index = _get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": source_index,
        },
        crs="EPSG:32632",
    )
    if split_points:
        return [points.clip(reg) for reg in region]
    else:
        return points.clip(region)


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


def xy_to_point(xy: tuple) -> shapely.Point:
    """Quick converter for xy tuples to shapely points.

    :param xy: The tuple with (x,y)
    :type xy: tuple
    :return: The point.
    :rtype: shapely.Point
    """
    pnt = shapely.Point(xy)
    return pnt


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
    old_crs = gdf.crs
    if old_crs != crs:
        gdf = gdf.to_crs(crs)
    gdf_buf = gdf.buffer(buffer_size)
    gdf_buf = gdf_buf.to_crs(old_crs)
    return gdf_buf


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
            for kk in shp_data.keys()
        ]
    )
    return buff_gdf
