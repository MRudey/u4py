"""
Contains several functions to perform spatial operations, such as reprojecting
or spatial lookup of features.
"""

from __future__ import annotations

import logging
import os
import pickle as pkl
import re
from typing import Callable, Iterable, List, Tuple

import geopandas as gp
import mahotas.polygon as mhpoly
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
from skimage import measure as skmeasure
from tqdm import tqdm

import u4py.io.files as u4files


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
                    x = int(fname[2:5]) * 1000
                    y = int(fname[6:]) * 1000
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
    if isinstance(bounds, mplax.Axes):
        xlim = bounds.get_xlim()
        ylim = bounds.get_ylim()
        poly = bounds_to_polygon((xlim[0], ylim[0], xlim[1], ylim[1]))
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
        "sums": [],  # Sum of all displacements (total volume balance)
        "vol_removed": [],  # Sum of all negative displacements (material removed)
        "vol_added": [],  # Sum of all positive displacements (material added)
        "vol_moved": [],  # Absolute sum of all displacements (volume moved)
        "avg_displ": [],  # Average displacement over whole area.
        "min": [],  # min displacement
        "max": [],  # max displacement
        "prc95": [],  # 95 percentile displacement
        "prc5": [],  # 5 percentile displacement
        "prc68": [],  # 68 percentile displacement
        "prc32": [],  # 32 percentile displacement
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
                                f"{levels[ii]} - {levels[ii+1]}"
                            )
                        else:
                            data["polygon_levels"].append(f">{levels[ii]}")
                        data["color_levels"].append(levels[ii])
                        data["areas"].append(pgon.area)
                        data["sums"].append(
                            calculate_in_contour(zz, p, np.nansum)
                        )
                        data["min"].append(
                            calculate_in_contour(zz, p, np.nanmin)
                        )
                        data["max"].append(
                            calculate_in_contour(zz, p, np.nanmax)
                        )
                        data["prc95"].append(
                            calculate_in_contour(zz, p, np.nanpercentile, q=95)
                        )
                        data["prc5"].append(
                            calculate_in_contour(zz, p, np.nanpercentile, q=5)
                        )
                        data["prc68"].append(
                            calculate_in_contour(zz, p, np.nanpercentile, q=68)
                        )
                        data["prc32"].append(
                            calculate_in_contour(zz, p, np.nanpercentile, q=32)
                        )
                        data["vol_removed"].append(
                            calculate_in_contour(zz, p, vol_removed)
                        )
                        data["vol_added"].append(
                            calculate_in_contour(zz, p, vol_added)
                        )
                        data["vol_moved"].append(
                            data["vol_added"][-1] + data["vol_removed"][-1]
                        )
                        data["avg_displ"].append(
                            data["sums"][-1] / data["areas"][-1]
                        )

    logging.debug("Generating GeoDataFrame")
    gdf = gp.GeoDataFrame(
        data=data,
        crs=crs,
    )
    return gdf


def calculate_in_contour(
    zz: np.ndarray, polygon: Iterable, fnc: Callable, **kwargs
) -> float:
    """Extracts the values within the polygon from the image in `zz`and applies the function to it.

    :param zz: The numpy array from where to extract the data
    :type zz: np.ndarray
    :param polygon: The polygon used for slicing as a list of (x,y) tuples
    :type polygon: Iterable
    :param fnc: The function to be used for the data.
    :type polygon: Callable
    :param **kwargs: Additional keyword arguments passed to the function.
    :return: The result of the function.
    :rtype: float
    """
    contour_int = [(int(np.round(y)), int(np.round(x))) for x, y in polygon]
    mask = np.zeros_like(zz)
    mhpoly.fill_polygon(contour_int, mask)
    mask = mask.astype(bool)
    if np.isfinite(zz[mask]).any():  # prevents all nans
        if kwargs:
            return fnc(zz[mask], **kwargs)
        else:
            return fnc(zz[mask])
    else:
        return 0


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
    collev = gdf.color_levels.to_numpy()

    un_groups = np.unique(groups)

    data = dict()
    for k in gdf.keys():
        data[k] = []

    for grp in tqdm(un_groups, desc="Joining by group", leave=False):
        slc = np.squeeze(np.argwhere(groups == grp))
        if slc.shape:
            subset = np.squeeze(collev[slc])
            un_subset = np.unique(subset)
            if len(un_subset) < len(subset):
                to_merge = [
                    np.squeeze(slc[np.argwhere(ii == subset)])
                    for ii in un_subset
                ]
                for mrg in to_merge:
                    if mrg.shape:
                        data["geometry"].append(
                            shapely.MultiPolygon(
                                merge_geometries(
                                    gdf["geometry"][mrg]
                                ).geometry.to_list()
                            )
                        )
                        data["polygon_levels"].append(
                            gdf["polygon_levels"][mrg[0]]
                        )
                        data["color_levels"].append(
                            gdf["color_levels"][mrg[0]]
                        )
                        data["areas"].append(np.sum(gdf["areas"][mrg]))
                        data["sums"].append(np.sum(gdf["sums"][mrg]))
                        data["vol_removed"].append(
                            np.sum(gdf["vol_removed"][mrg])
                        )
                        data["vol_added"].append(np.sum(gdf["vol_added"][mrg]))
                        data["vol_moved"].append(np.sum(gdf["vol_moved"][mrg]))
                        data["avg_displ"].append(
                            data["sums"][-1] / data["areas"][-1]
                        )
                        data["min"].append(np.nanmin(gdf["min"][mrg]))
                        data["max"].append(np.nanmax(gdf["max"][mrg]))
                        data["prc95"].append(np.nanmean(gdf["prc95"][mrg]))
                        data["prc5"].append(np.nanmean(gdf["prc5"][mrg]))
                        data["prc68"].append(np.nanmean(gdf["prc68"][mrg]))
                        data["prc32"].append(np.nanmean(gdf["prc32"][mrg]))
                        data["groups"].append(gdf["groups"][mrg[0]])

                    else:
                        for k in gdf.keys():
                            data[k].append(gdf[k][mrg])

            else:
                for ii in slc:
                    for k in gdf.keys():
                        data[k].append(gdf[k][ii])

        else:
            for k in gdf.keys():
                data[k].append(gdf[k][slc])
    return gp.GeoDataFrame(data=data, crs=gdf.crs)


def merge_geometries(geometries: gp.GeoSeries) -> gp.GeoSeries:
    """Merges the geometries when they overlap in a buffer radius of 10 cm.

    :param geometries: The input geometries
    :type geometries: gp.GeoSeries
    :return: The merged geometries
    :rtype: gp.GeoSeries
    """
    merged_geometries = []
    geom_buf = geometries.buffer(0.1).to_list()
    current_geom = geom_buf.pop(0)
    while len(geom_buf) > 0:
        overlapping = current_geom.overlaps(geom_buf)
        if np.any(overlapping):
            join_idx = [ii for ii, ovlp in enumerate(overlapping) if ovlp]
            to_join = [geom_buf[ii] for ii in join_idx]
            for ii in sorted(join_idx, reverse=True):
                del geom_buf[ii]
            to_join.append(current_geom)
            current_geom = gp.GeoSeries(to_join).unary_union
        else:
            if isinstance(current_geom, shapely.MultiPolygon):
                current_geom = current_geom.geoms[0]
            merged_geometries.append(current_geom)
            current_geom = geom_buf.pop(0)
    return gp.GeoSeries(merged_geometries)


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
    sub_set = get_subset(shp_gdf, group)
    try:
        sub_set_hull = gp.GeoDataFrame(
            geometry=[sub_set.unary_union.convex_hull.buffer(buffer_size)],
            crs=shp_gdf.crs,
        )
    except AttributeError:
        sub_set_hull = gp.GeoDataFrame()
    return sub_set_hull


def calculate_slope_in_shapes(
    shapes: gp.GeoDataFrame, tiff_folder: os.PathLike
) -> gp.GeoDataFrame:
    """Calculates the slope for each shape in `shapes` using the data from the DEM found in `tiff_folder`. Also computes some basic statistics for each area.

    :param shapes: The shapes in a geodataframe.
    :type shapes: gp.GeoDataFrame
    :param tiff_folder: The folder where to find the DEM data (in `*.adf` format.)
    :type tiff_folder: os.PathLike
    :return: A set of Polygons including mean and median slope, and standard deviation.
    :rtype: gp.GeoDataFrame
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = select_points_region(shapes, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    slope_data = {
        "geometry": [],
        "slope_mean": [],
        "slope_median": [],
        "slope_std": [],
    }
    for geom in shapes.geometry.to_list():
        if isinstance(geom, shapely.Polygon):
            data = []
            for fp in file_list:
                slope = compute_for_raster_in_geom(fp, geom, dem_slope)
                data.extend(slope[np.isfinite(slope)])
            slope_data["geometry"].append(geom)
            slope_data["slope_mean"].append(np.mean(data))
            slope_data["slope_median"].append(np.median(data))
            slope_data["slope_std"].append(np.std(data))
        elif isinstance(geom, shapely.MultiPolygon):
            for poly in list(geom.geoms):
                data = []
                for fp in file_list:
                    slope = compute_for_raster_in_geom(fp, poly, dem_slope)
                    data.extend(slope[np.isfinite(slope)])
                slope_data["geometry"].append(poly)
                slope_data["slope_mean"].append(np.mean(data))
                slope_data["slope_median"].append(np.median(data))
                slope_data["slope_std"].append(np.std(data))
    slopes = gp.GeoDataFrame(data=slope_data, crs=shapes.crs)
    return slopes


def compute_for_raster_in_geom(
    fpath: os.PathLike, geom: shapely.Polygon, func: Callable
) -> np.ndarray:
    """Applies the function `func` to the data in the raster file, masked by `geom`.

    :param fpath: The path to the raster file
    :type fpath: os.PathLike
    :param geom: The geometry to use for masking.
    :type geom: shapely.Polygon
    :param func: The function to apply to the data.
    :type func: Callable
    :return: The results of the function call.
    :rtype: np.ndarray
    """
    with rio.open(fpath, "r") as src:
        data, _ = riomask.mask(src, [geom], nodata=np.nan)
    return func(data[0])


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


def vol_added(im_data: np.ndarray) -> np.ndarray:
    """Calculates the volume added inside the area of the numpy array.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The slope at each pixel in degrees.
    :rtype: np.ndarray
    """
    return np.nansum(im_data[im_data > 0])


def vol_removed(im_data: np.ndarray) -> np.ndarray:
    """Calculates the volume removed inside the area of the numpy array.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The slope at each pixel in degrees.
    :rtype: np.ndarray
    """
    return np.nansum(im_data[im_data < 0])


def vol_moved(im_data: np.ndarray) -> np.ndarray:
    """Calculates the volume moved inside the area of the numpy array.

    :param im_data: The input raster dem data.
    :type im_data: np.ndarray
    :return: The slope at each pixel in degrees.
    :rtype: np.ndarray
    """
    return np.nansum(np.abs(im_data))


def roundness(shapes: gp.GeoDataFrame) -> list:
    """Calculates the roundness of all polygons in shapes.

    :param shapes: A geodataframe with polygons.
    :type shapes: gp.GeoDataFrame
    :return: The roundness between 0=not round and 1=perfectly round.
    :rtype: list
    """
    peri = shapes.geometry.length
    area = shapes.area
    roundness = round((4 * np.pi * area) / (area) ** 2, 3)
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
                flattn.append(round((a - b) / a, 1))
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
