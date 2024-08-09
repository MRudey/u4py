"""
Contains some sqlite functions for working with gpkg files
"""

from __future__ import annotations

import copy
import logging
import multiprocessing.pool as mpp
import os
import pickle as pkl
import sqlite3
import struct
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Iterable, Tuple

import geopandas as gp
import numpy as np
import shapely
import utm
from osgeo import ogr
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.plotting.preparation as u4plotprep
import u4py.utils.config as u4config
import u4py.utils.convert as u4conv


def get_BBD_table_names(file_path: os.PathLike) -> list:
    """Gets all tables which start with:
    |    'Zeitreihe_',
    |    'Ost_West', or
    |    'vertikal'

    :param file_path: The path to the database.
    :type file_path: os.PathLike
    :return: A list of tables to get from the database.
    :rtype: list
    """
    logging.debug("Getting table names.")
    con = sqlite3.connect(file_path)
    cur = con.cursor()

    # Get all table names for ascending and descending data
    tables = [
        res[0]
        for res in cur.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'Zeitreihe_%'"
        )
    ]
    # Append vertical and east-west datasets
    tables.extend(
        [
            res[0]
            for res in cur.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'vertikal%'"
            )
        ]
    )
    tables.extend(
        [
            res[0]
            for res in cur.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'Ost_West%'"
            )
        ]
    )
    con.close()
    if not tables:
        logging.info("No Tables according to scheme found.")
    else:
        logging.info(f"Found {len(tables)} tables in {file_path}")
        for ii, t in enumerate(tables):
            logging.debug(f" {ii:03g}: {t}")
    return tables


def get_table_names(
    file_path: os.PathLike, include_gpkg: bool = False
) -> list:
    """Gets a list of all table names in the sql file. Excludes gpkg specific
    tables by default!.

    :param file_path: The path to the sql file.
    :type file_path: os.PathLike
    :param include_gpkg: Whether to include specifics for gpkg files and rtrees, defaults to False
    :type include_gpkg: bool
    :return: The list of all tables.
    :rtype: list
    """
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    tables = [
        res[0]
        for res in cur.execute(
            "SELECT name FROM sqlite_schema WHERE type='table'"
        )
    ]
    con.close()
    if not include_gpkg:
        tables = [
            tab
            for tab in tables
            if not tab.startswith("rtree")
            and not tab.startswith("gpkg")
            and not tab.startswith("sqlite")
        ]
    return tables


def map_queries(queries: list) -> list:
    """Maps a list of sql queries to a parallel processing pool.

    :param queries: The list of queries as strings.
    :type queries: list
    :return: The results of the queries as a list.
    :rtype: list
    """
    """"""
    logging.info("Starting parallel sql extraction.")
    with Pool(u4config.cpu_count) as p:
        results = list(p.map(multi_proc_query, queries))
    return results


def multi_proc_query(args: Tuple) -> Any:
    """Multiprocessing wrapper for sql queries

    :param args: Input arguments
    :type args: Tuple
    :return: The result of the query.
    :rtype: Any
    """
    return single_query(*args)


def single_query(
    file_path: os.PathLike, query: str, jj: int = -1, n_expected: int = 0
) -> Any | Tuple[Any, int]:
    """Executes a single sql query for the given db-file.

    :param file_path: The path to the database.
    :type file_path: os.PathLike
    :param query: The sql query to execute.
    :type query: str
    :param jj: The number of the query (useful for Iterables), defaults to -1
    :type jj: int, optional
    :param n_expected: The number of expected outputs of the query. Useful to collect multiple outputs, defaults to 0 (single value)
    :type n_expected: int, optional
    :return: The result of the query.
    :rtype: Any
    """
    logging.debug(f"{query}")
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    if n_expected > 0:
        result = [value[:n_expected] for value in cur.execute(query)]
    else:
        result = [value[0] for value in cur.execute(query)]
    con.close()
    if jj >= 0:
        return (result, jj)
    else:
        return result


def table_to_dict(
    file_path: os.PathLike,
    table: str,
    bounds: Tuple = (),
    get_timeseries: bool = True,
    recalculate_stats: bool = False,
) -> dict:
    """Opens the given sql database and gets all content of the given table.

    :param file_path: The path to the database.
    :type file_path: os.PathLike
    :param table: The Table to get from.
    :type table: str
    :param bounds: Extent of a region where to get the data, limiting the number of SQL queries. The order follows the definition in geopandas: (`minx`, `miny`, `maxx`, `maxy`).
    :type bounds: Tuple
    :param get_timeseries: Whether to read the time series or not. Only loads the mean velocity and variance when False, defaults to True.
    :type get_timeseries: bool
    :param recalculate_stats: Whether to recalculate the mean and variance for the timeseries.
    :type recalculate_stats: bool
    :return: The content of the table.
    :rtype: dict

    This function has differently shaped and named dictionaries depending on
    the type of table or file.
    """
    # Get number of rows and names of columns
    num_points = get_num_entries(file_path, table)
    if num_points == 0:
        return

    logging.info(f"Opening {file_path} and getting content of {table}")
    # Read data
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    info = read_info(cur, table)
    info["num_points"] = num_points

    # Getting column names for coordinates
    # (BBD: X,Y,Z; EGMS: easting, northing, height)

    if "X" in info["all_keys"]:
        x_str = "X"
        y_str = "Y"
        z_str = "Z"
    elif "easting" in info["all_keys"]:
        x_str = "easting"
        y_str = "northing"
        z_str = "height"

    if len(bounds) > 0:
        if isinstance(bounds, tuple) or isinstance(bounds, list):
            where = (
                f"{x_str} > {bounds[0]} AND "
                + f"{x_str} < {bounds[2]} AND "
                + f"{y_str} > {bounds[1]} AND "
                + f"{y_str} < {bounds[3]} "
            )
        elif isinstance(bounds, gp.pd.DataFrame):
            where = (
                f"{x_str} > {bounds.minx.values[0]} AND "
                + f"{x_str} < {bounds.maxx.values[0]} AND "
                + f"{y_str} > {bounds.miny.values[0]} AND "
                + f"{y_str} < {bounds.maxy.values[0]}"
            )
        else:
            TypeError("Bounds of invalid type.")
    else:
        where = ""

    logging.info("Querying coordinates and keys")
    xx, yy, zz, ps_id = multi_col_select(
        cur,
        [f"{x_str}", f"{y_str}", f"{z_str}", info["id_key"]],
        table,
        where=where,
    )

    if info["has_time"] and get_timeseries:
        time, timeseries = read_timeseries(cur, table, info, where=where)
    if recalculate_stats:
        mean_vel, var_mean_vel = get_stats(cur, table, info, where, file_path)
    else:
        logging.info(f"Getting means.")
        mv_key, var_mv_key = get_meanvelo_keys(info["all_keys"])
        mean_vel, var_mean_vel = multi_col_select(
            cur, [mv_key, var_mv_key], table, where
        )
    con.close()

    if info["has_time"] and get_timeseries:
        output = {
            "x": np.array(xx),
            "y": np.array(yy),
            "z": np.array(zz),
            "time": time,
            "ps_id": np.array(ps_id),
            "timeseries": timeseries,
            "mean_vel": mean_vel,
            "var_mean_vel": var_mean_vel,
            "num_points": info["num_points"],
        }
    else:
        output = {
            "x": np.array(xx),
            "y": np.array(yy),
            "z": np.array(zz),
            "ps_id": np.array(ps_id),
            "mean_vel": np.array(mean_vel),
            "var_mean_vel": np.array(var_mean_vel),
            "num_points": info["num_points"],
        }

    return output


def get_stats(
    cur: sqlite3.Cursor,
    table: str,
    info: dict,
    where: str,
    file_path: os.PathLike,
) -> Tuple[np.ndarray, np.ndarray]:
    folder, file = os.path.split(file_path)
    stat_file_path = os.path.join(folder, file.replace(".gpkg", "_stats.pkl"))
    if os.path.exists(stat_file_path):
        logging.info("Loading Stats from pickle.")
        with open(stat_file_path, "rb") as pklfile:
            mean_vel, var_mean_vel = pkl.load(pklfile)
    else:
        logging.info("Recalculating stats.")
        time, timeseries = read_timeseries(cur, table, info, where=where)
        mean_vel, var_mean_vel = np.array(
            u4plotprep.get_linfit_each_timeseries(
                {"time": time, "timeseries": timeseries}
            )
        )
        with open(stat_file_path, "wb") as pklfile:
            pkl.dump((mean_vel, var_mean_vel), pklfile)

    return mean_vel, var_mean_vel


def select(
    cur: sqlite3.Cursor, column: str, table: str, where: str = ""
) -> Iterable:
    """Wrapper for a `SELECT` query

    :param cur: The cursor of the open database.
    :type cur: sqlite3.Cursor
    :param column: The column or statement to query.
    :type column: str
    :param table: The table where to extract
    :type table: str
    :param where: Predicates on rows, defaults to ""
    :type where: str, optional
    """

    if where:
        result = [
            value[0]
            for value in cur.execute(
                f"SELECT {column} FROM '{table}' WHERE {where}"
            )
        ]
    else:
        result = [
            value[0]
            for value in cur.execute(f"SELECT {column} FROM '{table}'")
        ]
    return result


def multi_col_select(
    cur: sqlite3.Cursor, columns: list[str], table: str, where: str = ""
) -> Iterable:
    """Wrapper for a `SELECT` query with multiple columns

    :param cur: The cursor of the open database.
    :type cur: sqlite3.Cursor
    :param column: A list of columns to query.
    :type column: list
    :param table: The table where to extract
    :type table: str
    :param where: Predicates on rows, defaults to ""
    :type where: str, optional
    """
    result = [list() for ii in range(len(columns))]
    cols = ""
    for cc in columns:
        cols += cc + ","
    cols = cols[:-1]
    if where:
        for value in cur.execute(
            f"SELECT {cols} FROM '{table}' WHERE {where}"
        ):
            for ii, val in enumerate(value):
                result[ii].append(val)
    else:
        for value in cur.execute(f"SELECT {cols} FROM '{table}'"):
            for ii, val in enumerate(value):
                result[ii].append(val)
    return result


def read_info(cur: sqlite3.Cursor, table: str) -> dict:
    """Reads some additional information from the sql tale

    Additional information are:

        - Non-Time Keys
        - The projected and geographical coordinate reference system
        - Whether the dataase contains time.
        - PS ID

    :param cur: The current cursor that accepts queries
    :type cur: sqlite3.Cursor
    :param table: The table from where to get most of the information.
    :type table: str
    :return: A dictionary with some additional info.
    :rtype: dict
    """
    logging.info("Reading Info from tables")
    info = dict()
    info["all_keys"] = [
        res[1] for res in cur.execute(f"PRAGMA TABLE_INFO({table})")
    ]

    spatial_ref = [
        int(res[0])
        for res in cur.execute("SELECT srs_id FROM gpkg_spatial_ref_sys")
    ]
    organiz = [
        res[0]
        for res in cur.execute("SELECT organization FROM gpkg_spatial_ref_sys")
    ]
    info["geo_crs"] = f"{organiz[2]}:{spatial_ref[2]}"
    info["proj_crs"] = f"{organiz[3]}:{spatial_ref[3]}"

    # Define type of file:
    info["has_time"] = True
    if "stack_ID" in info["all_keys"]:
        info["has_time"] = False
    elif "PS_ID" in info["all_keys"]:  # ASCE and DESC Data
        info["non_time_keys"] = ["X", "Y", "Z", "PS_ID", "Shape", "OBJECTID"]
        info["id_key"] = "PS_ID"
    if "Input" in info["all_keys"]:  # L3 Data
        filter_keys = [
            "OBJECTID",
            "Shape",
            "ID",
            "Input",
            "X",
            "Y",
            "Z",
            "mean_velo_vert",
            "var_mean_velo_vert",
            "mean_velo_east",
            "var_mean_velo_east",
        ]
        info["non_time_keys"] = [
            kk for kk in filter_keys if kk in info["all_keys"]
        ]
        info["id_key"] = "ID"
    # If we have no id key yet and the dataset is a EGMS dataset:
    if not "id_key" in info.keys() and "EGMS" in table:
        info["id_key"] = "pid"
        info["non_time_keys"] = info["all_keys"][:13]
    return info


def read_timeseries(
    cur: sqlite3.Cursor, table: str, info: dict, where: str = ""
) -> Tuple[np.ndarray, list]:
    """
    Generates queries for extracting time series data from the given file_path.
    These can be used with sqlite3 to read them from the tables directly.

    :param file_path: The path to the gpkg file
    :type file_path: os.PathLike
    :param table: The table/direction which to extract.
    :type table: str
    :param info: The extracted metadata from the gpkg file.
    :type info: dict
    :return: The timestamps and queries to extract.
    :rtype: Tuple[np.ndarray, list]
    """
    logging.debug(f"Getting timeseries")
    time_columns = [
        k for k in info["all_keys"] if k not in info["non_time_keys"]
    ]
    time = np.array([u4conv.sql_key_to_time(k) for k in time_columns])
    # cols = "'"
    # for cc in time_columns:
    #     cols += cc + "','"
    # cols = cols[:-2]
    if where:
        timeseries = cur.execute(
            f"SELECT * FROM '{table}' WHERE {where}"
        ).fetchall()
    else:
        timeseries = cur.execute(f"SELECT * FROM '{table}'").fetchall()

    timeseries = np.array(
        [r[len(info["non_time_keys"]) :] for r in timeseries]
    )

    timeseries[timeseries == None] = np.nan

    return time, timeseries.astype("float64")


def gen_timeseries_queries(
    file_path: os.PathLike, table: str, info: dict, where: str = ""
) -> Tuple[np.ndarray, list]:
    """
    Generates queries for extracting time series data from the given file_path.
    These can be used with sqlite3 to read them from the tables directly.

    :param file_path: The path to the gpkg file
    :type file_path: os.PathLike
    :param table: The table/direction which to extract.
    :type table: str
    :param info: The extracted metadata from the gpkg file.
    :type info: dict
    :return: The timestamps and queries to extract.
    :rtype: Tuple[np.ndarray, list]
    """
    logging.debug(f"Getting timeseries")
    key_list = [k for k in info["all_keys"] if k not in info["non_time_keys"]]
    time = np.array([u4conv.sql_key_to_time(k) for k in key_list])
    if where:
        queries = [
            (file_path, f"SELECT {k} FROM '{table}' WHERE {where}", jj)
            for jj, k in enumerate(key_list)
        ]
    else:
        queries = [
            (file_path, f"SELECT {k} FROM '{table}'", jj)
            for jj, k in enumerate(key_list)
        ]
    return time, queries


def load_tables(file_path: os.PathLike) -> dict:
    """Loads content of all tables in the given sql database and returns as a data dictionary.

    :param file_path: The input database.
    :type file_path: os.PathLike
    :return: The output dictionary.
    :rtype: dict
    """
    data = dict()
    tables = get_BBD_table_names(file_path)
    for table in tqdm(tables, desc="Reading from tables"):
        data[table] = table_to_dict(file_path, table)
    return data


def load_osm_gpkg(
    gpkg_file: os.PathLike,
    fclass: list = [],
    pool: bool | mpp.Pool = False,
    table_name: str = "",
) -> list:
    """
    Loads geometry data by reading the bytestream directly from the gpkg file.

    :param gpkg_file: The path to the gpkg file
    :type gpkg_file: os.PathLike
    :param fclass: Feature classes to extract, defaults to []
    :type fclass: list, optional
    :param pool: Use existing parallel Pool, defaults to False
    :type pool: bool, optional
    :param table_name: The name of the table where to extract the features, defaults to "".
    :type table_name: str, optional
    :return: A list containing all geometries
    :rtype: list
    """
    if not table_name:
        table_name = os.path.splitext(os.path.split(gpkg_file)[-1])[0]

    logging.info(f"Loading geometries from {gpkg_file}")
    con = sqlite3.connect(gpkg_file)
    cur = con.cursor()
    crs_query = cur.execute(
        "SELECT organization, srs_id FROM gpkg_spatial_ref_sys WHERE srs_id>0"
    ).fetchone()
    crs = f"{crs_query[0]}:{crs_query[1]}"
    if crs != "EPSG:4326":
        raise NotImplementedError("Only supports WGS 84 as input")
    query = f"SELECT geom FROM '{table_name}'"
    if fclass:
        where = ""
        for fc in fclass:
            if where:
                where += " or "
            where += f"fclass = '{fc}'"
        query += f" where {where}"
    geom_blobs = cur.execute(query).fetchall()
    con.close()
    limit = 3 * 10**5
    if len(geom_blobs) < limit and not pool:
        logging.info(f"Less than {limit} entries, non-parallel is faster.")
        geometries = [
            decode_geom(blob[0])
            for blob in tqdm(geom_blobs, desc="Decoding blobs", leave=False)
        ]
    else:
        geom_blobs = [blob[0] for blob in geom_blobs]
        if not pool:
            logging.info("Starting parallel pool")
            with Pool(u4config.cpu_count) as p:
                geometries = p.map(decode_geom, geom_blobs)
        else:
            logging.info("Using existing pool")
            geometries = pool.map(decode_geom, geom_blobs)

    return geometries


def decode_geom(stream: str) -> shapely.Geometry:
    """Primitive decoder for geometry blobs in a gpkg file. See http://www.geopackage.org./spec/#gpb_format.

    :param stream: The blob as a bytestring
    :type stream: str
    :return: The geometry geocoded in the data
    :rtype: shapely.Geometry

    The geometry blob contains a header, which may include the envelope of the features, and a well known binary (WKB) encoded geometry. We first decode the first 8 bytes to get some more information on what is stored in the blob:
        - 2 bytes: should be "GP" in ASCII
        - 1 byte: 8-bit unsigned Integer for version (0=v.1)
        - 1 byte: GeoPackageBinary flags byte -> has to be decoded to binary
        - 4 byte: 32-bit unsigned Integer with SRS ID.

    The GeoPackageBinary flag contains information about the length of the envelope that follows the first part of the header. This is needed to know where the WKB geometry starts. When we know that we can start to read the rest of the bytestring and feed it to `shapely.from_wkb` that creates the geometry.
    """
    first_header = struct.unpack("ccBcI", stream[:8])
    magic = first_header[0].decode() + first_header[1].decode()
    if magic == "GP":
        # version = first_header[2]
        flags = binary_flag(first_header[3])
        env_len = get_envlen(flags[-4:-1])
        srs_id = first_header[4]
        wkb_start = env_len + 8
        geometry = shapely.from_wkb(stream[wkb_start:])
        if srs_id == 4326:
            if geometry.geom_type == "Polygon":
                long, lat = geometry.exterior.coords.xy
                east, north, _, _ = utm.from_latlon(
                    np.array(lat), np.array(long)
                )
                points = [(e, n) for e, n in zip(east, north)]
                geometry = shapely.Polygon(points)
            elif geometry.geom_type == "Point":
                east, north, _, _ = utm.from_latlon(geometry.y, geometry.x)
                geometry = shapely.Point(east, north)
            elif geometry.geom_type == "LineString":
                long, lat = geometry.xy
                east, north, _, _ = utm.from_latlon(
                    np.array(lat), np.array(long)
                )
                points = [(e, n) for e, n in zip(east, north)]
                geometry = shapely.LineString(points)
            elif geometry.geom_type == "MultiPolygon":
                long, lat = geometry.envelope.exterior.coords.xy
                east, north, _, _ = utm.from_latlon(
                    np.array(lat), np.array(long)
                )
                points = [(e, n) for e, n in zip(east, north)]
                geometry = shapely.Polygon(points)
            else:
                NotImplementedError(
                    f"Conversion from {geometry.geom_type} not supported."
                )
        else:
            NotImplementedError("Supports only conversion from WGS84")
        return geometry
    else:
        UnicodeDecodeError("Not a valid GPKG enconding")


def binary_flag(inp: str) -> str:
    """Input GeoPackageBinary flag as an encoded bytestring.

    :param inp: The bytestring (single byte)
    :type inp: str
    :return: The bytestring in binary representation
    :rtype: str
    """
    bin_flags = f"{int(inp.hex()):0>8b}"
    return bin_flags


def get_envlen(eee: str) -> int:
    """Gets the length of an envelope as specified in the 5 to 7th bit of the binary flag.

    :param eee: The three relevant bits from the binary flag
    :type eee: str
    :return: The length of the envelope
    :rtype: int
    """
    envlen = [0, 32, 48, 48, 64]
    return envlen[int(eee, base=3)]


def gen_queries_psi_gpkg(
    file_path: os.PathLike, direction: str = "vertikal"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Opens a gpkg and generates queries to extract the data limited to psi files

    :param file_path: The database as a gpkg file.
    :type file_path: os.PathLike
    :param direction: The direction to use, defaults to "vertikal"
    :type direction: str, optional
    :return: Some data and preformatted queries to extract the data from the database.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]
    """
    logging.info(f"Generating queries to extract {direction} from {file_path}")
    # Get number of rows and names of columns
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    info = read_info(cur, direction)
    # Get Coordinates
    xx = np.array(select(cur, "X", direction))
    yy = np.array(select(cur, "Y", direction))
    zz = np.array(select(cur, "Z", direction))

    if info["has_time"]:
        time, queries = gen_timeseries_queries(file_path, direction, info)

    con.close()

    return xx, yy, zz, time, queries, info


def get_meanvelo_keys(all_keys: list) -> Tuple[str, str]:
    """Looks for the right mean velocity keys and return them.

    :param all_keys: List of all keys in the table.
    :type all_keys: list
    :return: The keys for the mean and variance.
    :rtype: Tuple[str, str]
    """
    if "mean_velocity" in all_keys and "var_mean_velocity" in all_keys:
        return ("mean_velocity", "var_mean_velocity")
    elif "mean_velo_vert" in all_keys and "var_mean_velo_vert" in all_keys:
        return ("mean_velo_vert", "var_mean_velo_vert")
    elif "mean_velo_east" in all_keys and "var_mean_velo_east" in all_keys:
        return ("mean_velo_east", "var_mean_velo_east")
    elif "mean_velocity" in all_keys and "mean_velocity_std" in all_keys:
        return ("mean_velocity", "mean_velocity_std")
    else:
        raise KeyError("No keys for mean velocity found.")


def get_num_entries(
    file_path: os.PathLike, table: str, region: shapely.Polygon | Tuple = ()
) -> int:
    """Gets number of entries in given sql file and table.

    :param file_path: The path to the sql file.
    :type file_path: os.PathLike
    :param table: The table to look for data.
    :type table: str
    :param region: The region to use for clipping the database
    :type region: str
    :return: The number of entries.
    :rtype: int
    """
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    if region:
        if isinstance(region, shapely.Polygon):
            region = region.bounds

        num_points = cur.execute(
            f"SELECT COUNT(*) FROM '{table}' WHERE "
            + f"X > {region[0]} AND X < {region[2]} AND "
            + f"Y > {region[1]} AND Y < {region[3]} "
        ).fetchone()[0]
    else:
        try:
            num_points = cur.execute(
                f"SELECT feature_count FROM gpkg_ogr_contents WHERE table_name='{table}'"
            ).fetchone()[0]
        except:
            num_points = cur.execute(
                f"SELECT COUNT(*) FROM '{table}'"
            ).fetchone()[0]
    con.close()
    return num_points


def get_subdivided_regions(
    file_path: os.PathLike, table: str, max_entries: int
) -> list[shapely.Polygon]:
    """Subdivides the regions into equally sized areas with a maximum number of entries. This is needed for keeping the shapefile size below 2GB.

    :param file_path: The path to the sql file.
    :type file_path: os.PathLike
    :param table: The table to look for data.
    :type table: str
    :param max_entries: The maximum number of entries that a region may have.
    :type max_entries: int
    :return: A list of Polygons dividing the area.
    :rtype: list[shapely.Polygon]
    """
    logging.info("Subdividing Polygons")
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    bounds = cur.execute(
        f"SELECT min_x, min_y, max_x, max_y FROM gpkg_contents WHERE table_name=='{table}'"
    ).fetchall()[0]
    xy = cur.execute(f"SELECT X, Y FROM '{table}'").fetchall()
    x = np.zeros(len(xy))
    y = np.zeros_like(x)
    for ii in range(len(xy)):
        x[ii], y[ii] = xy[ii]
    new_regions = u4spatial.subdivide_polygon(
        u4spatial.bounds_to_polygon(bounds), overlap=100
    )
    ne_new_regions = [
        get_num_entries_numpy(reg.bounds, x, y) for reg in new_regions
    ]
    while max(ne_new_regions) > max_entries:
        old_regions = copy.copy(new_regions)
        old_ne = copy.copy(ne_new_regions)
        new_regions = []
        ne_new_regions = []
        for reg, ne in zip(old_regions, old_ne):
            if ne > max_entries:
                subdiv = u4spatial.subdivide_polygon(reg, overlap=100)
                new_regions.extend(subdiv)
                ne_new_regions.extend(
                    [get_num_entries_numpy(sub.bounds, x, y) for sub in subdiv]
                )
            else:
                new_regions.append(reg)
                ne_new_regions.append(ne)
    return new_regions


def get_num_entries_numpy(bounds: tuple, x: np.ndarray, y: np.ndarray) -> int:
    """Returns the number of entries in x,y that are within the bounds.

    :param bounds: The boundary as a tuple of xmin, ymin, xmax, ymax
    :type bounds: tuple
    :param x: The x coordinates.
    :type x: np.ndarray
    :param y: The y coordinates
    :type y: np.ndarray
    :return: The number of entries
    :rtype: int
    """

    all_scl = np.bitwise_and(
        np.bitwise_and(x >= bounds[0], x < bounds[2]),
        np.bitwise_and(y >= bounds[1], y <= bounds[3]),
    )
    return len(all_scl[all_scl])


def get_crs(gpkg_path: os.PathLike, table: str = "") -> list[str]:
    """Reads the spatial reference from the gpkg file.

    :param gpkg_path: The path to the gpkg file.
    :type gpkg_path: os.PathLike
    :return: The spatial reference(s).
    :rtype: list[str] | str
    """
    con = ogr.Open(gpkg_path)
    if table:
        query = con.ExecuteSQL(
            f"SELECT srs_id from gpkg_geometry_columns WHERE table_name='{table}'"
        )
    else:
        query = con.ExecuteSQL(f"SELECT srs_id from gpkg_geometry_columns")
    srs_id = [layer.srs_id for layer in query]
    con = None
    return [f"EPSG:{srs}" for srs in srs_id]


def get_tables(gpkg_path: os.PathLike) -> list[str]:
    """Reads the tables from the gpkg file.

    :param gpkg_path: The path to the gpkg file.
    :type gpkg_path: os.PathLike
    :return: The table name(s).
    :rtype: list[str]
    """
    con = ogr.Open(gpkg_path)
    query = con.ExecuteSQL(f"SELECT table_name from gpkg_geometry_columns")
    tables = [layer.table_name for layer in query]
    con = None
    return tables


def ogr_spatial_select(
    gpkg_path: os.PathLike, table: str, wkt_bounds: str
) -> dict:
    """Selects all features from the given gpkg file in the table within the bounds.

    :param gpkg_path: The path to the gpkg file.
    :type gpkg_path: os.PathLike
    :param table: The table where to extract the data.
    :type table: str
    :param wkt_bounds: The boundary in well-known-text representation.
    :type wkt_bounds: str
    :return: A dictionary to create a geodataframe with all the geometries.
    :rtype: dict
    """
    # Get some info from the file
    geom_col = get_geometry_column(gpkg_path, table)[0]

    # Formulate query
    sql_query = (
        f"SELECT * FROM '{table}' "
        + "WHERE ST_Intersects("
        + f"ST_GeomFromText('{wkt_bounds}', 0), {geom_col}"
        + ")"
    )

    # Get the data
    con = ogr.Open(gpkg_path)
    logging.info("Querying for features")
    query = con.ExecuteSQL(sql_query)
    data = dict()
    for layer in query:
        geom = layer[geom_col]
        if geom:  # Sometimes the geometries are empty...?
            if not data:
                for field in layer.keys():
                    data[field] = []
                data["geometry"] = []
            for field in layer.keys():
                data[field].append(layer[field])
            data["geometry"].append(shapely.from_wkt(geom.ExportToWkt()))
    con = None
    return data


def get_geometry_column(gpkg_path: os.PathLike, table: str = "") -> list[str]:
    """Gets the geometry column name for the specified table.

    :param gpkg_path: The path to the gpkg file.
    :type gpkg_path: os.PathLike
    :param table: The table to query.
    :type table: str
    :return: The column name for the geometry.
    :rtype: str
    """
    con = ogr.Open(gpkg_path)
    if table:
        query = con.ExecuteSQL(
            f"SELECT column_name FROM gpkg_geometry_columns WHERE table_name='{table}'"
        )
    else:
        query = con.ExecuteSQL(
            f"SELECT column_name FROM gpkg_geometry_columns"
        )
    column_name = [layer.column_name for layer in query]
    con = None
    return column_name


def ogr_where_select(gpkg_path: os.PathLike, table: str, where: str) -> dict:
    """Selects all features from the given gpkg file in the table where `where` clause is True.

    :param gpkg_path: The path to the gpkg file.
    :type gpkg_path: os.PathLike
    :param table: The table where to extract the data.
    :type table: str
    :param where: The where clause for selection
    :type where: str
    :return: A dictionary to create a geodataframe with all the geometries.
    :rtype: dict
    """

    # Get some info from the file
    geom_col = get_geometry_column(gpkg_path, table)[0]

    # Formulate query
    sql_query = f"SELECT * FROM '{table}' WHERE {where}"

    # Get the data
    con = ogr.Open(gpkg_path)
    logging.info("Querying for features")
    query = con.ExecuteSQL(sql_query)
    data = dict()
    for layer in query:
        geom = layer[geom_col]
        if geom:  # Sometimes the geometries are empty...?
            if not data:
                for field in layer.keys():
                    data[field] = []
                data["geometry"] = []
            for field in layer.keys():
                data[field].append(layer[field])
            data["geometry"].append(shapely.from_wkt(geom.ExportToWkt()))
    con = None
    return data


def get_unique_entries(
    file_path: os.PathLike, field: str, tables: list[str] = []
) -> list[str]:
    """Gets all unique entries from the field for each table.

    :param file_path: The path to the sql database.
    :type file_path: os.PathLike
    :param field: The field name to get the values from.
    :type field: str
    :param tables: A single table, or list of table names, defaults to [] (all tables.)
    :type tables: str | list[str], optional
    :return: A list of unique entries from the tables.
    :rtype: list[str]
    """
    if not tables:
        tables = get_table_names(file_path)

    result = []
    for table in tables:
        res = single_query(
            file_path, f'SELECT DISTINCT "{field}" FROM "{table}"'
        )
        if len(res) == 1:
            if res[0] == field:
                if table == "lan-con":
                    res = "construction"
                elif table == "par-sur":
                    res = "parking"
        result.extend(res)

    return list(np.unique(result))


def get_bounds(file_path: os.PathLike) -> Tuple | list[tuple]:
    """Gets the bounds of all datasets in the given gpkg file.

    :param file_path: The path to the gpkg file.
    :type file_path: os.PathLike
    :param tables: A single table, or list of table names, defaults to [] (all tables.)
    :type tables: str, list[str]
    :return: A list of tuples containing (minx, miny, maxx, maxy) from the tables.
    :rtype: list[tuple]
    """

    result = single_query(
        file_path,
        "SELECT min_x,min_y,max_x,max_y FROM gpkg_contents",
        n_expected=4,
    )

    return result
