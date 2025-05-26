"""
Functions for working with geopackages. Has some overlap with functions from
the `sql` module as gpkg files are sqlite3 databases with some specific
additions.
"""

import logging
import os
from typing import Tuple

import geopandas as gp
import shapely
from osgeo import ogr

import u4py.analysis.spatial as u4spatial
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.utils.convert as u4conv
from u4py.utils.types import ShapeCfgDict

ogr.UseExceptions()


def load_and_buffer_gpkg(
    gpkg_path: os.PathLike,
    tiff_path: os.PathLike,
    tables: list = [],
    shp_cfg: ShapeCfgDict = {},
    out_crs: str = "EPSG:32632",
    buffer_dist: bool = False,
) -> gp.GeoDataFrame:
    """Loads all features within the bounds of a tiff file as a geodataframe.

    :param gpkg_path: The path to the gpkg file.
    :type gpkg_path: os.PathLike
    :param tiff_path: The path to the tiff file.
    :type tiff_path: os.PathLike
    :param tables: List of tables to extract, defaults to [] = all tables are extracted.
    :type tables: list, optional
    :param fclass_filter: List of feature classes to extract, defaults to [] = all fclasses are extracted.
    :type fclass_filter: list, optional
    :param out_crs: CRS for the output, defaults to "EPSG:32632".
    :type out_crs: str, optional
    :param buffer: Dictionary of buffer sizes for all `fclass_filter`, defaults to {} = no buffering.
    :type buffer: dict, optional
    :return: The loaded geometries in a geodataframe.
    :rtype: gp.GeoDataFrame
    """

    logging.debug(f"Loading Features in {os.path.split(tiff_path)[-1]}")
    if shp_cfg and tables:
        UserWarning("Shape config overwrites tables argument!")
    if shp_cfg:
        tables = [
            val.replace(".shp", "") for val in shp_cfg["shp_file"].values()
        ]
        keys = [k for k in shp_cfg["shp_file"].keys()]

    # Use all tables in case none were given
    if not tables:
        tables = u4sql.get_table_names(gpkg_path)

    geometries = []
    fclasses = []
    buffer_dists = []

    # Loop over all tables
    con = ogr.Open(gpkg_path)
    for ii, table in enumerate(tables):
        # See if we need to filter specific fclasses and set buffer distance
        if shp_cfg:
            fclass_filter = shp_cfg["fclass"][keys[ii]]
            buffer_dist = shp_cfg["buffer_dist"][keys[ii]]
        else:
            fclass_filter = []

        # Cache for geometries, depending on CRS we need to reproject them.
        geometries_table = []

        # Getting spatial reference of geometries in table and calculate
        # boundaries accordingly
        srs_query = con.ExecuteSQL(
            f"SELECT srs_id from gpkg_geometry_columns WHERE table_name == '{table}'"
        )
        srs_id = srs_query[0].srs_id
        wkt_bounds = u4tiff.wkt_from_tiff_bounds(tiff_path, f"EPSG:{srs_id}")

        # Load geometries within bounds
        layer = con.ExecuteSQL(
            f"SELECT * FROM '{table}' WHERE ST_Intersects(geom, ST_GeomFromText('{wkt_bounds}', 0))",
        )

        # Set fclass type
        for feature in layer:
            if hasattr(feature, "fclass"):
                fclass = feature.fclass
            elif hasattr(feature, "generator_"):
                fclass = "power"
            elif hasattr(feature, "man_made"):
                fclass = feature.man_made
            elif table == "lan-con":
                fclass = "construction"
            elif table == "par-sur":
                fclass = "parking"
            elif hasattr(feature, "manual_class_1"):
                fclass = feature.manual_class_1

            # Collect features in list
            if not fclass_filter or fclass in fclass_filter:
                fclasses.append(fclass)
                if buffer_dist:
                    buffer_dists.append(buffer_dist)
                geometries_table.append(
                    shapely.from_wkt(feature.geom.ExportToWkt())
                )

        # If CRS/SRS of current table does not match with WGS84,
        # reproject the geometries.
        if geometries_table:
            if srs_id != 4326:
                geometries_table = (
                    gp.GeoDataFrame(
                        geometry=geometries_table, crs=f"EPSG:{srs_id}"
                    )
                    .to_crs("EPSG:4326")
                    .geometry
                )
        geometries.extend(geometries_table)
    con = None  # Closing dataset for GDAL<3.8

    logging.debug("Creating new geodataframe from extracted geometries")
    gdf = gp.GeoDataFrame(
        crs="EPSG:4326",
        data={"fclass": fclasses, "geometry": geometries},
    ).to_crs(out_crs)

    if buffer_dists:
        logging.debug("Buffering the geometries")
        buf_geoms = [
            geom.buffer(buffer_dists[ii])
            for ii, geom in enumerate(gdf.geometry)
        ]
        gdf = gp.GeoDataFrame(
            crs=gdf.crs,
            data={"fclass": fclasses, "geometry": buf_geoms},
        )
    return gdf


def load_gpkg_data_point(
    point: list | Tuple | gp.GeoDataFrame | shapely.Point,
    radius: float,
    gpkg_file_path: os.PathLike,
    table: str = "vertikal",
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> Tuple[dict, gp.GeoDataFrame]:
    """Selects PSI measurements in a `radius` around the specified `point` from the given **GPKG** File.

    This function loads the data directly using sql, no second step is required.

    :param point: The center point of the query.
    :type point: list
    :param radius: The radius to calculate the buffer
    :type radius: float
    :param gpkg_file_path: The file or folder to select from
    :type gpkg_file_path: os.PathLike
    :param table: The table from which to extract the data, defaults to "vertikal"
    :type table: str, optional
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The points from `psi_file_path` in a `radius` round `point` and the region.
    :rtype: Tuple[dict, gp.GeoDataFrame]
    """
    gpkg_crs = u4sql.get_crs(gpkg_file_path, table)[0]
    region = u4spatial.region_around_point(point, radius, crs=gpkg_crs)

    points = load_gpkg_data_region_ogr(
        region, gpkg_file_path, table, crs=region.crs
    )
    data = u4conv.reformat_gdf_to_dict(points)

    return data, region


def load_gpkg_data_osm(
    osm_query: dict,
    gpkg_file_path: os.PathLike,
    table: str = "vertikal",
    split_points: bool = False,
    crs: str = "EPSG:32632",
) -> Tuple[dict, gp.GeoDataFrame]:
    """Selects PSI measurements in a region defined by an OSM query from the given **GPKG** File.

    This function loads the data directly using sql, no second step is required.

    :param osm_query: A properly formatted osm query in dictionary form. (see https://osmnx.readthedocs.io/en/stable/ for more)
    :type osm_query: dict
    :param gpkg_file_path: The file or folder to select from
    :type gpkg_file_path: os.PathLike
    :param table: The table from which to extract the data, defaults to "vertikal"
    :type table: str, optional
    :param split_points: Splits the output into a list of points based on the selection, defaults to False
    :type split_points: bool, optional
    :param crs: The coordinate system of the output points, defaults to "EPSG:32632".
    :type crs: str, optional
    :return: The points from `psi_file_path` within the query region and the region
    :rtype: Tuple[dict, gp.GeoDataFrame]
    """

    region = u4spatial.get_osm_region(osm_query, crs=crs)

    data = u4sql.table_to_dict(gpkg_file_path, table, bounds=region.bounds)
    clipped_data = u4spatial.clip_data_points(
        data, region, split_points=split_points, crs=crs
    )
    if clipped_data:
        clipped_data["crs"] = crs
    return clipped_data, region


def load_gpkg_data_region(
    region: gp.GeoDataFrame | gp.GeoSeries,
    gpkg_file_path: os.PathLike,
    table: str = "",
    split_points: bool = False,
    gpkg_crs: str = "",
    region_crs: str = "EPSG:32632",
) -> dict:
    """Selects PSI measurements in the specified region from the given **GPKG** File.

    This function loads the data directly using sql, no second step is required.

    :param region: The region as GeoDataFrame
    :type region: gp.GeoDataFrame | gp.GeoSeries
    :param gpkg_file_path: The file or folder to select from
    :type gpkg_file_path: os.PathLike
    :param table: The table from which to extract the data, defaults to ""
    :type table: str, optional
    :param split_points: Whether to split the points into individual sets, defaults to False
    :type split_points: bool, optional
    :param gpkg_crs: The CRS of the coordinates in the tables of the gpkg, might be different than the CRS of the geometries which is given in the metadata, defaults to "".
    :type gpkg_crs: str, optional
    :param region_crs: The CRS of the region used for clipping, defaults to "EPSG:32632". Only has an effect if the region is not a GeoDataFrame.
    :type region_crs: str, optional
    :return: The points from `psi_file_path` in the `region`.
    :rtype: dict
    """
    if not table:
        tables = u4sql.get_table_names(gpkg_file_path)
        if len(tables) < 1:
            raise ValueError("File does not contain any tables")
        elif len(tables) == 1:
            table = tables[0]
        else:
            raise ValueError(
                f"File contains several tables, please specify appropriate table from: {tables}"
            )

    if not gpkg_crs:
        # Get CRS of sql datbase and convert region to it.
        gpkg_crs = u4sql.get_crs(gpkg_file_path, table)[0]
    if isinstance(region, gp.GeoDataFrame):
        if region.crs != gpkg_crs:
            region = region.to_crs(gpkg_crs)
    elif isinstance(region, (gp.pd.Series, gp.GeoSeries)):
        UserWarning(
            "Region for selection is a GeoSeries, check input CRS manually!"
        )
        region = gp.GeoDataFrame(
            geometry=[region.geometry], crs=region_crs
        ).to_crs(gpkg_crs)
    elif isinstance(region, shapely.Polygon):
        UserWarning(
            "Region for selection is a GeoSeries, check input CRS manually!"
        )
        region = gp.GeoDataFrame(geometry=[region], crs=region_crs).to_crs(
            gpkg_crs
        )
    # Extract Data
    data = u4sql.table_to_dict(gpkg_file_path, table, bounds=region.bounds)
    clipped_data = u4spatial.clip_data_points(
        data, region, split_points=split_points, crs=gpkg_crs
    )
    if clipped_data:
        clipped_data["crs"] = gpkg_crs
    return clipped_data


def load_gpkg_data_where(
    gpkg_file_path: os.PathLike, table: str = "", where: str = ""
) -> dict:
    """Selects data from a gpkg file where the given SQL statement is true.

    :param gpkg_file_path: The path to the gpkg file.
    :type gpkg_file_path: os.PathLike
    :param table: The table where to look for data, defaults to ""
    :type table: str, optional
    :param where: The SQL query for filtering, defaults to ""
    :type where: str, optional
    :return: The points from `psi_file_path`.
    :rtype: dict
    """
    if not table:
        tables = u4sql.get_table_names(gpkg_file_path)
        if len(tables) < 1:
            raise ValueError("File does not contain any tables")
        elif len(tables) == 1:
            table = tables[0]
        else:
            raise ValueError(
                f"File contains several tables, please specify appropriate table from: {tables}"
            )

    data = u4sql.table_to_dict(gpkg_file_path, table, where=where)
    return data


def load_gpkg_data_region_ogr(
    region: gp.GeoDataFrame | gp.GeoSeries | shapely.Polygon,
    gpkg_file_path: os.PathLike,
    table: str = "",
    crs: str = "EPSG:32632",
    clip: bool = True,
) -> gp.GeoDataFrame:
    """Uses OGR to read data from a table in a gpkg file within a specific region.

    :param region: The region to use for spatial selection.
    :type region: gp.GeoDataFrame | gp.GeoSeries
    :param gpkg_file_path: The path to the gpkg database.
    :type gpkg_file_path: os.PathLike
    :param table: The table where to find the data, defaults to ""
    :type table: str, optional
    :param crs: The coordinate system of the output, defaults to "EPSG:32632"
    :type crs: str, optional
    :return: The data within the selected region.
    :rtype: gp.GeoDataFrame
    """

    # Check tables
    if not table:
        tables = u4sql.get_tables(gpkg_file_path)
        if len(tables) < 1:
            raise ValueError("File does not contain any tables")
        elif len(tables) == 1:
            table = tables[0]
        else:
            raise ValueError(
                f"File contains several tables, please specify appropriate table from: {tables}"
            )

    # Make region if a polygon was given (has to be of same crs as gpkg)
    gpkg_crs = u4sql.get_crs(gpkg_file_path, table)[0]
    if isinstance(region, shapely.Polygon):
        region = gp.GeoDataFrame(geometry=[region], crs=gpkg_crs)
    # Homogenize input CRS for extraction
    elif region.crs != gpkg_crs:
        region = region.to_crs(gpkg_crs)

    # Get WKT representation of geometry and load data
    if "geometry" in region.keys():
        region_wkt = region["geometry"].to_wkt().to_list()[0]
    else:
        region_wkt = region.geometry.to_wkt().to_list()[0]
    data = u4sql.ogr_spatial_select(gpkg_file_path, table, region_wkt)
    if data:
        try:
            if clip:
                gdf = gp.GeoDataFrame(data, crs=gpkg_crs).clip(region)
            else:
                gdf = gp.GeoDataFrame(data, crs=gpkg_crs)
            gdf = gdf.to_crs(crs)
        except:
            input_reg = gp.GeoDataFrame(data, crs=gpkg_crs).buffer(0)
            data["geometry"] = input_reg
            if clip:
                gdf = gp.GeoDataFrame(data, crs=gpkg_crs).clip(region)
            else:
                gdf = gp.GeoDataFrame(data, crs=gpkg_crs)
            gdf = gdf.to_crs(crs)
        return gdf
    else:
        return []


def load_gpkg_data_where_ogr(
    gpkg_file_path: os.PathLike,
    table: str = "",
    where: str = "",
    crs: str = "EPSG:32632",
) -> gp.GeoDataFrame:
    """Uses OGR to read data from a table in a gpkg file with a specific where clause.

    :param gpkg_file_path: The path to the gpkg database.
    :type gpkg_file_path: os.PathLike
    :param table: The table where to find the data, defaults to ""
    :type table: str, optional
    :param where: The where clause for selection
    :type where: str
    :return: The shapes where the `where` clause is true.
    :rtype: gp.GeoDataFrame
    """

    # Check tables
    if not table:
        tables = u4sql.get_tables(gpkg_file_path)
        if len(tables) < 1:
            raise ValueError("File does not contain any tables")
        elif len(tables) == 1:
            table = tables[0]
        else:
            raise ValueError(
                f"File contains several tables, please specify appropriate table from: {tables}"
            )

    data = u4sql.ogr_where_select(gpkg_file_path, table, where)
    if data:
        gpkg_crs = u4sql.get_crs(gpkg_file_path, table)[0]
        gdf = gp.GeoDataFrame(data, crs=gpkg_crs)
        gdf = gdf.to_crs(crs)
        return gdf
    else:
        return []
