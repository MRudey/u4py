import os
import pickle as pkl
from os import PathLike
from typing import Tuple

import geopandas as gp
import h5py
import numpy as np
import osmnx
import rasterio as rio
import rasterio.warp as riowarp
import scipy.spatial as spspatial
import shapely


def get_features(in_dict: dict, features: list):
    """Gets specied features from input dictionary"""
    ind = []
    for feat in features:
        ind.extend(np.argwhere(in_dict["fclass"] == feat))
    coords = np.squeeze(in_dict["geometry"][ind])
    names = [
        str((n.encode("latin_1")).decode("utf8"))
        for n in np.squeeze(in_dict["name"][ind])
    ]

    return (names, coords)


def reproject_raster(in_path, out_path, output_crs):
    """reproject raster to project crs"""
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


def get_cKDTree(file_path):
    """
    Loads all x and y coordinates from the given file or folder and returns a
    cKDTree for easy spatial lookup.
    """
    coords = get_coords(file_path)

    return spspatial.cKDTree(coords)


def get_coords(input_data: PathLike | list) -> list:
    """
    Loads all x and y coordinates from the given file or folder.

    Works with single hdf or pkl files as well as folders with many files.
    """
    coords = None

    if isinstance(input_data, list):
        if isinstance(input_data[0], str):
            coords = file_list_to_coords(input_data)
        elif isinstance(input_data[0], tuple):
            coords = tuple_list_to_coords(input_data)

    elif isinstance(input_data, str):
        if input_data.endswith(".h5"):
            coords = h5_to_coords(input_data)
        elif input_data.endswith(".pkl"):
            coords = pkl_to_coords(input_data)

    if coords is None:
        raise NotImplementedError(
            "Conversion of this type to lookup table not implemented!"
        )
    return coords


def file_list_to_coords(input_data: list) -> list:
    """
    Converts a file list of h5 files with x and y coordinates in their names
    to a list of coordinates.
    """
    coords = [
        shapely.Point(
            int(f[f.find("_x") + 2 : f.find("_y")]),
            int(f[f.find("_y") + 2 : f.find(".h5")]),
        )
        for f in os.listdir(input_data)
        if f.endswith(".h5")
    ]
    return coords


def h5_to_coords(h5file_path: os.PathLike) -> list:
    """Converts a h5 file containing x and y to list of coordinates"""
    with h5py.File(h5file_path, "r") as h5file:
        coords = np.array(
            [(x, y) for x, y in zip(h5file["x"][()], h5file["y"][()])]
        )
    return coords


def tuple_list_to_coords(tuple_list: list) -> list:
    """
    Converts a list of tuples from a loaded pkl file to list of coordinates."""
    coords = np.array([(d[0], d[1]) for d in tuple_list])
    return coords


def pkl_to_coords(pkl_path: os.PathLike) -> list:
    """
    Converts contents of a pickled inversion results file to list of
    coordinates.
    """
    with open(pkl_path, "rb") as pklfile:
        data = pkl.load(pklfile)[0]
    coords = tuple_list_to_coords(data)
    return coords


def spatial_lookup(
    input_feature, points: gp.GeoDataFrame, n: int = 1
) -> list[Tuple[float, int]]:
    """Does a spatial lookup for the nearest point to all points in `points`.
    The data at `input_feature` has to have a valid format for creating a  lookup Tree.

    Arguments:
        input_feature -- A variable that contains some sort of x and y table
          which can be converted to a lookup table with `get_cKDTree()`
        points -- The points to query.

    Keyword Arguments:
        n -- The number of nearest neighbors (default: {1}).

    Returns:
        A list of `n` nearest neighbors for each point in the form of (distance, index)
    """
    lut = get_cKDTree(input_feature)
    points = points.values
    closest = [lut.query((p[1].x, p[1].y), n) for p in points]
    return closest


def select_points_osm(query, psi_file_path):
    """
    Selects points from the specified file or files in folder
    and crops them by the rectangles found in the given osm query.
    """

    osm_data = osmnx.geometries_from_address(
        query["address"], tags=query["tags"]
    ).to_crs("EPSG:32632")

    coords = get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": np.arange(len(coords)),
        },
        crs="EPSG:32632",
    )
    return points.clip(osm_data)


def select_points_region(region, psi_file_path):
    """
    Selects points from the specified file and crops them by the rectangles
    found in the given region.
    """
    if region.crs != "EPSG:32632":
        region = region.to_crs("EPSG:32632")
    coords = get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": np.arange(len(coords)),
        },
        crs="EPSG:32632",
    )
    return points.clip(region)


def select_points_point(point, radius, psi_file_path):
    region = gp.GeoDataFrame(
        {"geometry": [shapely.Point(point).buffer(radius)]}, crs="EPSG:32632"
    )
    coords = get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": coords,
            "source_index": np.arange(len(coords)),
        },
        crs="EPSG:32632",
    )
    return points.clip(region)


def get_rois(file_path: os.PathLike) -> list[Tuple[str, gp.GeoDataFrame]]:
    regions = gp.read_file(file_path)
    rois = [(r[1], r[2]) for r in regions.values]
    return rois
