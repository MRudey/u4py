"""
Contains several functions to perform spatial operations, such as reprojecting
or spatial lookup of features.
"""

from __future__ import annotations

import itertools
import logging
import operator as op
import os
import pickle as pkl
import re
from typing import Callable, List, Tuple

import geopandas as gp
import matplotlib.axes as mplax
import numpy as np
import osmnx
import rasterio as rio
import rasterio.coords
import rasterio.mask as riomask
import rasterio.warp as riowarp
import scipy.spatial as spspatial
import shapely
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.point import Point
from shapely.errors import GEOSException
from skimage import measure as skmeasure
from tqdm import tqdm

from u4py.utils.types import ShapeCfgDict


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


def get_cKDTree(file_path: os.PathLike) -> spspatial.KDTree:
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

    if isinstance(in_path, list) and len(in_path) > 0:
        if isinstance(in_path[0], str):
            coords, source_index = _file_list_to_coords(in_path)
        elif isinstance(in_path[0], tuple):
            coords = _tuple_list_to_coords(in_path)
            source_index = np.arange(len(coords))

    elif isinstance(in_path, str):
        if in_path.endswith(".pkl"):
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
                if fname.startswith("g_"):
                    if (  # Check if folder contains adf files
                        len(
                            [
                                f
                                for f in os.listdir(fpath)
                                if f.endswith(".adf")
                            ]
                        )
                        > 0
                    ):
                        logging.debug(
                            "Using extents of adf dataset for coordinates."
                        )
                        with rio.open(fpath) as raster:
                            coords.append(bounds_to_polygon(raster.bounds))
                        source_index.append(ii)
                    else:
                        logging.debug("Using folder name for coordinates.")
                        coord_str = re.findall("g_(\d*)_(\d*)", fname)[0]
                        if len(coord_str) < 1:
                            raise NameError("Unkown tiff naming scheme.")
                        x = int(coord_str[0]) * 1000
                        y = int(coord_str[1]) * 1000
                        coords.append(
                            shapely.Polygon(
                                [
                                    (x, y),
                                    (x + 5000, y),
                                    (x + 5000, y + 5000),
                                    (x, y + 5000),
                                    (x, y),
                                ]
                            )
                        )
                        source_index.append(ii)
                else:
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
            "source_ind": source_index,
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
    :param crs: The CRS of thinput region, defaults to ""
    :type crs: str, optional
    :return: The points from `psi_file_path` cropped to the `region`.
    :rtype: gp.GeoDataFrame
    """

    coords, source_index = _get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_ind": source_index,
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
    psi_file_path: os.PathLike | List[os.PathLike],
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
    :type psi_file_path: os.PathLike | List[os.PathLike]
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
            "source_ind": source_index,
        },
        crs=crs,
    )

    if split_points:
        return [points.clip(reg) for reg in region]
    else:
        return points.clip(region)


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
    if "mean_vel" in data.keys():
        points = points.assign(mean_vel=data["mean_vel"])
        points = points.assign(var_mean_vel=data["var_mean_vel"])

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
    if len(clipped_data) > 0:
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
        if "mean_vel" in clipped_data.keys():
            loaded_data["mean_vel"] = clipped_data["mean_vel"]
            loaded_data["var_mean_vel"] = clipped_data["var_mean_vel"]
    else:
        loaded_data = dict()
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


def bounds_to_polygon(
    bounds: rasterio.coords.BoundingBox | Tuple | mplax.Axes,
) -> shapely.Polygon:
    """Creates a `shapely.Polygon` from the `BoundingBox` of loaded raster, `tuple` or `matplotlib.axes.Axes`.

    :param bounds: The `BoundingBox` of a raster, a set of (minx,miny,maxx, maxy) coordinates, or a `matplotlib.axes.Axes` object.
    :type bounds: rasterio.coords.BoundingBox | Tuple  | matplotlib.axes.Axes
    :return: The polygon.
    :rtype: shapely.Polygon
    """
    if isinstance(bounds, rasterio.coords.BoundingBox):
        poly = shapely.Polygon(
            shell=(
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom),
                (bounds.left, bounds.bottom),
            )
        )
    elif isinstance(bounds, mplax.Axes):
        xlim = bounds.get_xlim()
        ylim = bounds.get_ylim()
        poly = bounds_to_polygon((xlim[0], ylim[0], xlim[1], ylim[1]))
    elif isinstance(bounds, gp.pd.Series):
        poly = shapely.Polygon(
            shell=(
                (bounds.minx, bounds.miny),
                (bounds.minx, bounds.maxy),
                (bounds.maxx, bounds.maxy),
                (bounds.maxx, bounds.miny),
                (bounds.minx, bounds.miny),
            )
        )
    else:
        poly = shapely.Polygon(
            shell=(
                (bounds[0], bounds[1]),
                (bounds[0], bounds[3]),
                (bounds[2], bounds[3]),
                (bounds[2], bounds[1]),
                (bounds[0], bounds[1]),
            )
        )
    return poly


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


def contour_shapes(
    coords: dict,
    zz: np.ndarray,
    levels: list,
    min_area: float,
    crs: str,
    delete_outside: bool = False,
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
    :param delete_outside: Whether to set values above the highest level to nan (and below the lowest), defaults to False
    :type delete_outside: bool, optional
    :return: A list of Polygons together with the levels they have been generated from.
    :rtype: gp.GeoDataFrame
    """
    logging.debug("Generating Contour Shapes")

    if delete_outside:
        logging.debug("Deleting values outside level range")
        zz[np.abs(zz) > levels[-1]] = np.nan
        zz[np.abs(zz) < levels[0]] = np.nan

    logging.debug("Making contour set")
    contours = [
        skmeasure.find_contours(np.rot90(zz, 3), level=level)
        for level in levels
    ]

    logging.debug("Separating contour set")
    data = {
        "geometry": [],  # The polygon shapes
        "polygon_levels": [],  # The levels of the polygon
        "color_levels": [],  # The midpoint of the levels for easy plotting
        "areas": [],  # The area of the polygon
    }
    for ii, segs in enumerate(contours):
        if len(segs) > 0:
            for p in segs:
                if len(p) > 3:
                    # Scale contours back to CRS
                    p_rescaled = [
                        (
                            coords["x"] + c[0] * coords["dx"],
                            coords["y"] - c[1] * coords["dy"],
                        )
                        for c in p
                    ]
                    pgon = shapely.geometry.Polygon(p_rescaled)
                    if pgon.area > min_area:
                        data["geometry"].append(pgon)
                        if ii < len(levels) - 1:
                            data["polygon_levels"].append(
                                f"{levels[ii]} - {levels[ii + 1]}"
                            )
                        else:
                            data["polygon_levels"].append(f">{levels[ii]}")
                        data["color_levels"].append(levels[ii])
                        data["areas"].append(pgon.area)
    if len(data["geometry"]) > 0:
        logging.debug("Generating GeoDataFrame")
        gdf = gp.GeoDataFrame(
            data=data,
            crs=crs,
        )
        return gdf
    else:
        return gp.GeoDataFrame()


def vol_removed(values: np.ndarray) -> float:
    """Sum of all negative displacements.

    :param values: Displacement in polygon
    :type values: np.ndarray
    :return: Sum of all negative input values
    :rtype: float
    """
    values = values[values < 0]
    if np.isfinite(values).any():  # prevents all nans
        return np.nansum(np.abs(values))
    else:
        return 0


def vol_added(values: np.ndarray) -> float:
    """Sum of all positive displacements.

    :param values: Displacement in polygon
    :type values: np.ndarray
    :return: Sum of all positive input values
    :rtype: float
    """
    values = values[values > 0]
    if np.isfinite(values).any():  # prevents all nans
        return np.nansum(np.abs(values))
    else:
        return 0


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


def subdivide_polygon(
    roi: gp.GeoDataFrame | shapely.Polygon,
    overlap: int = 250,
) -> list:
    """Splits a Polygon into two equally sized subpolygons with `overlap`.

    :param roi: The input polygon
    :type roi: gp.GeoDataFrame | shapely.Polygon
    :param overlap: The amount of overlap in meters, defaults to 250
    :type overlap: int, optional
    :return: A tuple with two new Polygons
    :rtype: list
    """
    logging.debug("Subdividing and getting number of entries.")
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

    return [new_poly1, new_poly2]


def subdivide_gdf_by_num(
    gdf: gp.GeoDataFrame, nsub: int = 2
) -> gp.GeoDataFrame:
    """Subdivides the boundaries of the input geodataframe along x and y `nsub`-times.

    :param gdf: The geodataframe for subdividing
    :type gdf: gp.GeoDataFrame
    :param nsub: The number of subdivisions, defaults to 2
    :type nsub: int, optional
    :return: A geodataframe containing polygons that subdivide the input geodataframe.
    :rtype: gp.GeoDataFrame
    """
    bounds = gdf.total_bounds
    xq = np.linspace(bounds[0], bounds[2], nsub + 1)
    yq = np.linspace(bounds[1], bounds[3], nsub + 1)
    geometries = []
    for ii in range(4):
        for jj in range(4):
            geometries.append(
                bounds_to_polygon((xq[ii], yq[jj], xq[ii + 1], yq[jj + 1]))
            )
    return gp.GeoDataFrame(geometry=geometries, crs=gdf.crs)


def subdivide_gdf_by_squares(
    gdf: gp.GeoDataFrame, width: float = 1000
) -> gp.GeoDataFrame:
    """Subdivides the boundaries of the input geodataframe into squares of `width` metres. The edges of the squares are not located along the edges of the input but rather overlap them.

    :param gdf: The geodataframe for subdividing
    :type gdf: gp.GeoDataFrame
    :param width: The width of the squares, defaults to 1000
    :type width: float, optional
    :return: A geodataframe containing polygons that split the input into equally sized squares.
    :rtype: gp.GeoDataFrame
    """
    in_crs = gdf.crs
    if in_crs != "EPSG:32632":
        gdf = gdf.to_crs("EPSG:32632")

    bounds = gdf.total_bounds
    minx = np.floor(bounds[0] / width) * width
    miny = np.floor(bounds[1] / width) * width
    maxx = np.ceil(bounds[2] / width) * width
    maxy = np.ceil(bounds[3] / width) * width

    xq = np.arange(minx, maxx + width, width)
    yq = np.arange(miny, maxy + width, width)
    geometries = []
    for ii in range(len(xq) - 1):
        for jj in range(len(yq) - 1):
            geometries.append(
                bounds_to_polygon((xq[ii], yq[jj], xq[ii + 1], yq[jj + 1]))
            )

    return gp.GeoDataFrame(geometry=geometries, crs="EPSG:32632").to_crs(
        in_crs
    )


def group_nearest(x: np.ndarray, y: np.ndarray, max_dist: float) -> list:
    """Groups input points based on their maximum distance between each other.

    :param x: The x coordinate of the points
    :type x: np.ndarray
    :param y: The y coordinate of the points
    :type y: np.ndarray
    :param max_dist: The maximum distance below which two points belong together in the same CRS as the points.
    :type max_dist: float
    :return: A numpy array of integers with the group for each point.
    :rtype: list
    """
    logging.info("Calculating distance matrix. USES A LOT OF RAM!")
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt((xx - xx.T) ** 2 + (yy - yy.T) ** 2)
    near_indices = [
        np.squeeze(np.argwhere(dist[ii, :] <= max_dist)).tolist()
        for ii in tqdm(range(len(x)), desc="Computing indices", leave=False)
    ]

    merged_idx = []
    for nidx in tqdm(near_indices, desc="Merging indices", leave=False):
        current = []
        get_near_idx(nidx, near_indices, current)
        merged_idx.append(str(np.unique(current).tolist()))
    unique_merges = np.unique(merged_idx)
    groups = [
        np.argwhere(mgd_idx == unique_merges)[0][0]
        for mgd_idx in tqdm(merged_idx, desc="Grouping merges", leave=False)
    ]
    logging.info(f"{len(unique_merges)} individual groups have been found")
    return groups


def get_near_idx(nidx: list | int, near_indices: list, current: list):
    """Adds nearest indices from the list and recurses through the entries
    that have not been added so far to `current`.

    :param nidx: A list of indices or single index of nearest neighbours.
    :type nidx: list | int
    :param near_indices: The whole list of nearest neighbours for recursion.
    :type near_indices: list
    :param current: The list of all nearest neighbours for the initial point. This gets bigger with each iteration.
    :type current: list
    """
    if isinstance(nidx, int) and nidx not in current:
        current.append(nidx)
    elif isinstance(nidx, list):
        for nii in nidx:
            if nii not in current:
                current.append(nii)
                get_near_idx(near_indices[nii], near_indices, current)


def join_groups_with_level(gdf: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Joins all geometries of the same level and group into a single (Multi-)
    Polygon.

    :param gdf: The input geodatabase with grouped items.
    :type gdf: gp.GeoDataFrame
    :return: A newly created geodatabase with the joined geometries
    :rtype: gp.GeoDataFrame
    """
    if not ("color_levels" in gdf.keys() and "groups" in gdf.keys()):
        raise KeyError(
            "The GDF is missing the required keys for merging: 'color_levels' and/or 'groups'."
        )

    groups = gdf.groups.to_numpy()

    data = dict()
    for k in gdf.keys():
        data[k] = []

    for grp in tqdm(np.unique(groups), desc="Joining by group", leave=False):
        grp_slc = np.argwhere(groups == grp)
        grp_slc = grp_slc.reshape(len(grp_slc))

        # If there is more than one contour of the same group
        if len(grp_slc) > 1:
            sub_group = gdf.iloc[grp_slc]
            colors_in_sub_group = sub_group.color_levels.to_numpy()
            for col in np.unique(colors_in_sub_group):
                col_slc = np.argwhere(colors_in_sub_group == col)
                col_slc = col_slc.reshape(len(col_slc))

                # If there is more than one contour per color level
                if len(col_slc) > 1:
                    merged_gs = merge_geometries(sub_group.iloc[col_slc])
                    for k in merged_gs.keys():
                        data[k].extend(merged_gs[k])

                # Only for single color level contours
                else:
                    for k in sub_group.keys():
                        data[k].append(sub_group[k].to_numpy()[col_slc][0])

        # Only for single contour groups
        else:
            for k in gdf.keys():
                data[k].append(gdf[k].to_numpy()[grp_slc][0])

    return gp.GeoDataFrame(data=data, crs=gdf.crs)


def merge_geometries(geometries: gp.GeoSeries) -> dict:
    """Merges the geometries when they overlap in a buffer radius of 10 cm.

    :param geometries: The input geometries
    :type geometries: gp.GeoSeries
    :return: A dictionary of merged geometries
    :rtype: gp.GeoSeries
    """
    polygon_level = geometries.polygon_levels.iloc[0]
    color_level = geometries.color_levels.iloc[0]
    group = geometries.groups.iloc[0]
    logging.debug(f"Merging geometries of group {group}")
    merged_geometries = []

    geom_buf = geometries.buffer(0.1).to_list()
    current_geom = geom_buf.pop(0)
    while len(geom_buf) > 0:
        overlapping = current_geom.overlaps(geom_buf)  # 1
        if np.any(overlapping):
            # Get geometries to join
            join_idx = [ii for ii, ovlp in enumerate(overlapping) if ovlp]
            to_join = [geom_buf[ii] for ii in join_idx]
            # Remove them from list
            for ii in sorted(join_idx, reverse=True):
                del geom_buf[ii]
            # Add current geometry to joining list
            to_join.append(current_geom)
            # Join Geometries
            current_geom = gp.GeoSeries(to_join).unary_union
            # Start again looking for more geometries in the vicinity at #1

        else:  # No overlapping geometries found
            merged_geometries.append(current_geom)  # Add geometry to list
            current_geom = geom_buf.pop(0)  # Take next geometry
    merged_geometries.append(current_geom)

    data = {
        "geometry": merged_geometries,
        "polygon_levels": [polygon_level] * len(merged_geometries),
        "color_levels": [color_level] * len(merged_geometries),
        "groups": [group] * len(merged_geometries),
        "areas": [geom.area for geom in merged_geometries],
    }
    return data


def get_subset(
    shp_gdf: gp.GeoDataFrame, group: int, level: float = 0.5
) -> gp.GeoDataFrame:
    """Selects a subset of contours from the geodataframe. The selection is based on the `group` and +- `level`.

    :param shp_gdf: The full set of thresholded contours.
    :type shp_gdf: gp.GeoDataFrame
    :param group: The group number.
    :type group: str
    :param level: The level to select, defaults to 0.5 (the smallest detected contour.)
    :type level: float, optional
    :return: The selected subset from the main geodataframe.
    :rtype: gp.GeoDataFrame
    """
    logging.debug(f"Getting subset of group {group}")
    slc = (shp_gdf.groups == group) & (
        (shp_gdf.color_levels == -level) | (shp_gdf.color_levels == level)
    )
    return shp_gdf[slc]


def get_subset_hull(
    shp_gdf: gp.GeoDataFrame, group: int, buffer_size: float
) -> gp.GeoDataFrame:
    """Gets the hull of a subset. First selects the subset using the `group` and then calculates the convex hull of all selected shapes. The hull is then buffered by `buffer_size` for better coverage of the immediate surroundings.

    :param shp_gdf: The full set of thresholded contours.
    :type shp_gdf: gp.GeoDataFrame
    :param group: The group number.
    :type group: int
    :param buffer_size: The buffer size around the hull.
    :type buffer_size: float
    :return: The buffered hull of the subset.
    :rtype: gp.GeoDataFrame
    """
    logging.debug(f"Getting subset hull of group {group}")
    sub_set = get_subset(shp_gdf, group)
    try:
        if len(sub_set) > 0:
            sub_set_hull = gp.GeoDataFrame(
                geometry=[sub_set.unary_union.convex_hull.buffer(buffer_size)],
                crs=shp_gdf.crs,
            )
    except AttributeError:
        logging.info(
            f"Could not create hull for group {group}: AttributeError."
        )
        sub_set_hull = gp.GeoDataFrame()
    except GEOSException:
        logging.info(
            f"Could not create hull for group {group}: GEOS Exeption, "
            + "probably a topology error."
        )
        sub_set_hull = gp.GeoDataFrame()
    return sub_set_hull


def compute_for_raster_in_geom(
    fpath: os.PathLike, geom: shapely.Polygon, func: Callable, geom_crs: str
) -> np.ndarray | int:
    """
    Applies the function `func` to the data in the raster file, masked by
    `geom`. Checks if the shape is actually within the raster file to avoid
    warnings and returns `None` if the shape is outside.

    :param fpath: The path to the raster file
    :type fpath: os.PathLike
    :param geom: The geometry to use for masking.
    :type geom: shapely.Polygon
    :param func: The function to apply to the data.
    :type func: Callable
    :param geom_crs: The coordinate system of the input geometries.
    :type geom_crs: str
    :return: The results of the function call, returns 0 when the geometry lies outside the raster bounds.
    :rtype: np.ndarray | int
    """

    with rio.open(fpath, "r") as src:
        # In some cases the input crs of the raster dataset is not correct, so
        # we have to adjust our input geometries.
        crs_epsg = src.crs.to_epsg()
        if not crs_epsg or crs_epsg != geom_crs:
            geom_gdf = gp.GeoDataFrame(geometry=[geom], crs=geom_crs).to_crs(
                src.crs
            )
            geom = geom_gdf.geometry[0]
        if shape_in_tiff(geom, src):
            data, _ = riomask.mask(src, [geom], nodata=np.nan)
            return func(data[0])
        else:
            return None


def get_values(im_data: np.ndarray) -> np.ndarray:
    """
    Returns all numeric values within the image. This just extracts the values
    that are not nan from a masked raster using `compute_for_raster_in_geom`.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The masked values.
    :rtype: np.ndarray
    """
    return im_data[np.isfinite(im_data)]


def dem_slope(im_data: np.ndarray) -> np.ndarray:
    """Calculates the slope inside the area of the numpy array.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The slope at each pixel in degrees.
    :rtype: np.ndarray
    """
    px, py = np.gradient(im_data)
    slope = np.degrees(np.arctan(np.sqrt(px**2 + py**2)))
    return slope


def area_per_feature(
    gdf: gp.GeoDataFrame, index_column: str
) -> Tuple[list[str], list[float]]:
    """Computes the area for each unique entry of the given index column in the geodataframe.

    :param gdf: The input shapes with at least one index column for sorting.
    :type gdf: gp.GeoDataFrame
    :param index_colum: The column used for indexing.
    :type index_column: str
    :return: A list of all unique features and their area.
    :rtype: Tuple[list[str], list[float]]
    """
    areas = []
    fclasses = []
    if len(gdf) > 0:
        fclass_list = list(np.unique(gdf[index_column]))
        for lnd in fclass_list:
            area = round(gdf[gdf[index_column] == lnd].area.sum(), 1)
            if area >= 1:  # filters negligible areas
                areas.append(area)
                fclasses.append(lnd)
    return (fclasses, areas)


def vol_moved(im_data: np.ndarray) -> np.ndarray:
    """Calculates the volume moved inside the area of the numpy array.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The volume moved.
    :rtype: np.ndarray
    """
    if np.isfinite(im_data).any():
        return np.nansum(np.abs(im_data))
    else:
        return 0


def vol_error(im_data: np.ndarray) -> np.ndarray:
    """Calculates the error of volume estimates inside the area of the numpy array.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The error of volume estimates.
    :rtype: np.ndarray
    """
    if np.isfinite(im_data).any():
        return 0.3**2 * len(im_data[np.isfinite(im_data)])
    else:
        return 0


def roundness(shapes: gp.GeoDataFrame) -> list:
    """Calculates the roundness of all polygons in shapes.

    :math:`Roundness=\\frac{4\\pi A}{p^2}`

    :param shapes: A geodataframe with polygons.
    :type shapes: gp.GeoDataFrame
    :return: The roundness between 0=not round and 1=perfectly round.
    :rtype: list
    """
    peri = shapes.geometry.length
    area = shapes.area
    roundness = round((4 * np.pi * area) / (peri) ** 2, 3)
    return roundness.to_list()


def flattening(shapes: gp.GeoDataFrame) -> list:
    """Calculates the ellipticity of all polygons in shapes. Defined as:

    :math:`f=\\frac{a-b}{a}`

    with a and b as the semi-axes of an ellipse fit to the coordinates.

    :param shapes: A geodataframe with polygons.
    :type shapes: gp.GeoDataFrame
    :return: The ellipticity.
    :rtype: list
    """
    aa = []
    bb = []
    tt = []
    flattn = []
    for geom in shapes.geometry.to_list():
        if isinstance(geom, shapely.Polygon):
            xx, yy = geom.exterior.coords.xy
            xy = np.array([(x, y) for x, y in zip(xx.tolist(), yy.tolist())])
            ellipse = skmeasure.EllipseModel()
            success = ellipse.estimate(xy)
            if success:
                _, _, a, b, theta = ellipse.params
                aa.append(round(a, 2))
                bb.append(round(b, 2))
                tt.append(round(theta, 1))
                flattn.append(round((a - b) / a, 1))
        elif isinstance(geom, shapely.MultiPolygon):
            geom = list(geom.geoms)[0]
            xx, yy = geom.exterior.coords.xy
            xy = np.array([(x, y) for x, y in zip(xx.tolist(), yy.tolist())])
            ellipse = skmeasure.EllipseModel()
            success = ellipse.estimate(xy)
            if success:
                _, _, a, b, theta = ellipse.params

                aa.append(round(a, 2))
                bb.append(round(b, 2))
                tt.append(round(theta, 1))
                try:
                    flattn.append(round((a - b) / a, 1))
                except ZeroDivisionError:
                    flattn.append(0)
    return (aa, bb, tt, flattn)


def reverse_geolocate(gdf: gp.GeoDataFrame) -> list:
    locations = []
    geolocator = Nominatim(user_agent="u4py email=rudolf@geo.tu-darmstadt.de")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    centroids = gdf.to_crs("EPSG:4326").geometry.centroid.to_list()
    for point in tqdm(centroids, "Reverse Geolocation"):
        try:
            lon = point.x
            lat = point.y
            location = reverse(Point(lat, lon)).raw["display_name"]
        except:
            location = ""
        locations.append(location)
    return locations


def shape_in_tiff(shp: shapely.Polygon, src: rasterio.DatasetReader) -> bool:
    """Checks if the shape is within the bounds of the raster file.

    :param shp: The shape.
    :type shp: gp.Polygon
    :param src: The loaded raster dataset.
    :type src: rasterio.DatasetReader
    :return: True if the shape is within.
    :rtype: bool
    """

    rst_bnd = src.bounds
    shp_bnd = shp.bounds
    result = (
        # lies in x range
        (shp_bnd[0] >= rst_bnd.left and shp_bnd[0] <= rst_bnd.right)
        or (shp_bnd[2] >= rst_bnd.left and shp_bnd[2] <= rst_bnd.right)
    ) and (
        # lies in y range
        (shp_bnd[1] >= rst_bnd.bottom and shp_bnd[1] <= rst_bnd.top)
        or (shp_bnd[3] >= rst_bnd.bottom and shp_bnd[3] <= rst_bnd.top)
    )
    return result


def feret_diameters(polygon: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the minimum and maximum Feret diameters of a convex polygon
    based on the rotating caliper method

    :param polygon: A numpy array of (x, y) coordinates
    :type polygon: np.ndarray
    :return: The minimum and maximum Feret diameter
    :rtype: Tuple[float, float]
    """
    if np.all(polygon[0] == polygon[-1]):
        polygon = polygon[:-1]
    length = len(polygon)
    Ds = np.empty(length)

    for i in range(length):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % length]

        ds = np.abs(np.cross(p2 - p1, p1 - polygon) / np.linalg.norm(p2 - p1))

        Ds[i] = np.max(ds)
    minf = np.min(Ds)
    maxf = np.max(Ds)

    return (minf, maxf)


def get_clipped_road_area(
    roads: gp.GeoDataFrame,
    road_type: str,
    clip_reg: gp.GeoDataFrame,
    shp_cfg: ShapeCfgDict,
) -> gp.GeoDataFrame:
    """Buffers and clips the roads to the area of interest, removing tunnels as well.

    :param roads: The roads dataframe.
    :type roads: gp.GeoDataFrame
    :param road_type: The list of roads to use.
    :type road_type: str
    :param clip_reg: The region used for clipping.
    :type clip_reg: gp.GeoDataFrame
    :param shp_cfg: The shape config containing buffer sizes.
    :type shp_cfg: ShapeCfgDict
    :return: The clipped roads as polygons.
    :rtype: gp.GeoDataFrame
    """
    roads = roads[roads["tunnel"] == "F"]
    road_gdf = gp.GeoDataFrame(
        geometry=gp.pd.concat(
            [
                roads[roads["fclass"] == rds]
                for rds in shp_cfg["fclass"][road_type]
            ]
        ).buffer(shp_cfg["buffer_dist"][road_type]),
        crs=roads.crs,
    )

    return road_gdf.clip(clip_reg)


def road_info(
    roads_data: gp.GeoDataFrame, fclass: str
) -> Tuple[list] | Tuple[list, float]:
    """
    Calculates the length and names of the features in the geodataframe with
    the given fclass.

    :param roads_data: A set of roads or railways
    :type roads_data: gp.GeoDataFrame
    :param fclass: The feature class to filter for, e.g. "motorway"
    :type fclass: str
    :return: The road names and road lengths, for railways the name is empty and the length is a single float
    :rtype: Tuple[list] | Tuple[list, float]
    """
    if hasattr(roads_data, "ref"):
        road_names = []
        road_length = []
        for ii, road in roads_data[roads_data.fclass == fclass].iterrows():
            if road.ref not in road_names:
                road_names.append(road.ref)
                road_length.append(road.geometry.length)
            else:
                ridx = road_names.index(road.ref)
                road_length[ridx] += road.geometry.length
    elif hasattr(roads_data, "code"):
        road_names = []
        road_length = 0
        for ii, road in roads_data[roads_data.fclass == fclass].iterrows():
            road_length += road.geometry.length

    return road_names, road_length


def ckdnearest(
    geom_a: shapely.Point, gdf_b: gp.GeoDataFrame, col: list[str] = ["Place"]
) -> gp.GeoDataFrame:
    """Finds nearest line features in `gdf_b` to the points in `gdf_a`.

    :param gdf_a: A geodataframe containing points
    :type gdf_a: gp.GeoDataFrame
    :param gdf_b: A geodataframe containing lines
    :type gdf_b: gp.GeoDataFrame
    :param col: The column to select, defaults to ["Place"]
    :type col: list, optional
    :return: _description_
    :rtype: gp.GeoDataFrame
    """
    geom_a = (geom_a.x, geom_a.y)
    geom_b = [
        np.array(geom.coords) for geom in gdf_b.geometry.explode().to_list()
    ]
    idx_b = tuple(
        itertools.chain.from_iterable(
            [
                itertools.repeat(i, x)
                for i, x in enumerate(list(map(len, geom_b)))
            ]
        )
    )
    geom_b = np.concatenate(geom_b)
    ckd_tree = spspatial.cKDTree(geom_b)
    dist, idx = ckd_tree.query([geom_a], k=1)
    idx = op.itemgetter(*idx)(idx_b)
    return dist, idx


def get_nearest_road_segment(
    point: shapely.Point, road_data: gp.GeoDataFrame, fclass: str
) -> Tuple[str, int]:
    """Gets the closest road segment from `road_data` to the given `point`. Returns the name of the road and its name.

    :param point: The point to use for querying.
    :type point: shapely.Point
    :param road_data: A set of roads to select from.
    :type road_data: gp.GeoDataFrame
    :param fclass: The feature class to filter the selection.
    :type fclass: str
    :return: The name of the road and the distance in metres.
    :rtype: Tuple[str, int]
    """
    dist, idx = ckdnearest(
        point,
        road_data[road_data.fclass == fclass],
    )
    if hasattr(road_data, "name") and hasattr(road_data, "ref"):
        name = road_data[road_data.fclass == fclass].iloc[idx].ref
        if not name:
            name = road_data[road_data.fclass == fclass].iloc[idx].name
    else:
        name = ""
    return name, int(dist)
