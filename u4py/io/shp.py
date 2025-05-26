"""
Functions for working with shapefiles.
"""

import logging
import os
import sys
import time
from typing import Tuple

import fiona
import geopandas as gp
import geopandas.io.file as gpiofile
import numpy as np
import osmnx
import pyproj
import shapely
from fiona import drivers as fiona_env
from packaging.version import Version
from tqdm import tqdm

import u4py.utils.utils as u4utils
from u4py.utils.types import U4Project


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
        in_data, in_crs = fiona_load(clipped_path)
        clipped_data = gp.GeoDataFrame(geometry=in_data, crs=in_crs)

    else:
        logging.info("Creating new clipped shapefile")
        in_data, in_crs = fiona_load(file_path_in, fclasses=fclass)
        in_gdf = gp.GeoDataFrame(geometry=in_data, crs=in_crs)
        logging.info("Clipping Data")
        if in_crs != mask.crs:
            mask = mask.to_crs(in_gdf.crs)
        clipped_data = in_gdf.clip(mask)
        logging.info("Saving clipped data.")
        clipped_data.to_file(clipped_path)
    return clipped_data, clipped_path


def get_osm_as_shp(
    tags: dict,
    project: U4Project,
    query: str = "Hesse",
    shape_type: shapely.GeometryType = shapely.Point,
    overwrite: bool = False,
) -> gp.GeoDataFrame:
    """Loads point data from openstreetmap in the `query` region with the `tags`. Saves results in a shape file.

    :param tags: The tags for open street map (same as for the overpass API)
    :type tags: dict
    :param project: The loaded project file (for path management)
    :type project: U4Project
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


def gdb_to_shp(gdb_path: os.PathLike) -> list[os.PathLike]:
    """Converts all features inside a geodatabase to shapefiles with the same name.

    :param gdb_path: The path to the geodatabase.
    :type gdb_path: os.PathLike
    :return: A list of paths to the shapefiles.
    :rtype: list[os.PathLike]
    """
    base_path, gdb_name = os.path.split(gdb_path)
    gdb_name = gdb_name.replace(".gdb", "")
    export_path = os.path.join(base_path, gdb_name + "_shpfiles")
    os.makedirs(export_path, exist_ok=True)
    shp_list = []
    layers = fiona.listlayers(gdb_path)
    for layer in tqdm(layers, desc="Extracting Shapes", leave=False):
        gdf = gp.read_file(gdb_path, layer=layer)
        gdf = clean_fields(gdf)
        out_path = os.path.join(export_path, layer + ".shp")
        gdf.to_file(out_path)
        shp_list.append(out_path)
    return shp_list


def clean_fields(gdf: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Cleans all unsupported fields from the geodataframe and converts them to corresponding types.

    :param gdf: The input dataframe
    :type gdf: gp.GeoDataFrame
    :return: The cleaned dataframe
    :rtype: gp.GeoDataFrame
    """

    for k in gdf.keys():
        if gdf[k].dtype == "datetime64[ns, UTC]":
            gdf[k] = gp.pd.Series([str(df) for df in gdf[k]])

    return gdf


def to_file_fiona(
    df: gp.GeoDataFrame,
    filename: os.PathLike,
    driver: str,
    schema: dict = None,
    crs: str = "",
    mode: str = "w",
    chunk_size: int = 1000,
    position: int = None,
    **kwargs,
):
    """Monkey patched version of the geopandas `_to_file_fiona` version including a tqdm progressbar to see the saving progress.

    :param df: The source geodataframe
    :type df: gp.GeoDataFrame
    :param filename: The filename where to save the data
    :type filename: os.PathLike
    :param driver: The driver for the file
    :type driver: str
    :param schema: The schema, defaults to None
    :type schema: dict
    :param crs: The CRS, defaults to ""
    :type crs: str
    :param mode: The mode of the file, defaults to "w" (write).
    :type mode: str
    """
    if schema is None:
        schema = gpiofile.infer_schema(df)

    if crs:
        crs = pyproj.CRS.from_user_input(crs)
    else:
        crs = df.crs

    with fiona_env():
        crs_wkt = None
        try:
            gdal_version = Version(
                fiona.env.get_gdal_release_name().strip("e")
            )  # GH3147
        except (AttributeError, ValueError):
            gdal_version = Version("2.0.0")  # just assume it is not the latest
        if gdal_version >= Version("3.0.0") and crs:
            crs_wkt = crs.to_wkt()
        elif crs:
            crs_wkt = crs.to_wkt("WKT1_GDAL")
        with fiona.open(
            filename,
            mode=mode,
            driver=driver,
            crs_wkt=crs_wkt,
            schema=schema,
            **kwargs,
        ) as colxn:
            # colxn.writerecords(df.iterfeatures())
            records = []
            tic = time.time()
            for ii, feature in enumerate(df.iterfeatures()):
                if ii and (not (ii % chunk_size) or ii == len(df)):
                    if position:
                        u4utils.eta(tic, ii / len(df))
                        logging.info(f"#{position} : {ii}/{len(df)}")
                    colxn.writerecords(records)
                    records = []
                else:
                    records.append(feature)
                # colxn.write(feature)


def fiona_load(shp_path: os.PathLike, fclasses: list = []) -> gp.GeoDataFrame:
    logging.info("Reading with fiona.")
    with fiona.open(shp_path, "r") as shapefile:
        if "fclass" in shapefile.keys() and fclasses:
            geometries = [
                shapely.geometry.shape(feature.geometry)
                for feature in tqdm(
                    shapefile,
                    desc="Reading selected shapes from file",
                    leave=False,
                    file=sys.stdout,
                )
                if feature.properties["fclass"] in fclasses
            ]
        else:
            geometries = [
                shapely.geometry.shape(feature.geometry)
                for feature in tqdm(
                    shapefile,
                    desc="Reading all shapes from file",
                    leave=False,
                    file=sys.stdout,
                )
            ]
        crs = shapefile.crs
    return geometries, crs
