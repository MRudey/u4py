"""
Contains some sqlite functions for working with gpkg files
"""
from __future__ import annotations

import itertools
import logging
import multiprocessing.pool as mpp
import os
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

import u4py.utils.config as u4config


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
        results = list(
            tqdm(
                p.map(multi_proc_query, queries),
                total=len(queries),
                desc="Processing queries",
                leave=False,
            )
        )
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
    file_path: os.PathLike, query: str, jj: int = -1
) -> Any | Tuple[Any, int]:
    """Executes a single sql query for the given db-file.

    :param file_path: The path to the database.
    :type file_path: os.PathLike
    :param query: The sql query to execute.
    :type query: str
    :param jj: The number of the query (useful for Iterables), defaults to -1
    :type jj: int, optional
    :return: The result of the query.
    :rtype: Any
    """
    logging.debug(f"{query}")
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    result = [value[0] for value in cur.execute(query)]
    con.close()
    if jj >= 0:
        return (result, jj)
    else:
        return result


def sql_key_to_time(key: str) -> datetime:
    """Returns a datetime object for the given date in sql format.

    :param key: The date as given in the database.
    :type key: str
    :return: A datetime object of the string.
    :rtype: datetime
    """
    return datetime.strptime(key, "date_%Y%m%d")


def table_to_dict(
    file_path: os.PathLike,
    table: str,
    bounds: Tuple = (),
    get_timeseries: bool = True,
) -> dict:
    """Opens the given sql database and gets all content of the given table.

    :param file_path: The path to the database.
    :type file_path: os.PathLike
    :param table: The Table to get from.
    :type table: str
    :param bounds: Extent of a region where to get the data, limiting the number of SQL queries. The order follows the definition in geopandas: (`minx`, `miny`, `maxx`, `maxy`).
    :type bounds: Tuple
    :param get_timeseries: Whether to read the time series or not. Only loads the mean velocity and variance when False, defaults to True.
    :type get_timeseries: str
    :return: The content of the table.
    :rtype: dict

    This function has differently shaped and named dictionaries depending on
    the type of table or file.
    """
    logging.info(f"Opening {file_path} and getting content of {table}")
    # Get number of rows and names of columns
    con = sqlite3.connect(file_path)
    cur = con.cursor()
    num_points = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if num_points == 0:
        return
    info = read_info(cur, table)
    info["num_points"] = num_points

    if len(bounds) > 0:
        if isinstance(bounds, tuple) or isinstance(bounds, list):
            where = (
                f"X > {bounds[0]} AND X < {bounds[2]} "
                + f"Y > {bounds[1]} AND Y < {bounds[3]} "
            )
        elif isinstance(bounds, gp.pd.DataFrame):
            where = (
                f"X > {bounds.minx.values[0]} AND "
                + f"X < {bounds.maxx.values[0]} AND "
                + f"Y > {bounds.miny.values[0]} AND "
                + f"Y < {bounds.maxy.values[0]}"
            )
        else:
            TypeError("Bounds of invalid type.")
    else:
        where = ""

    logging.info("Querying coordinates and keys")
    xx, yy, zz, ps_id = multi_col_select(
        cur, ["X", "Y", "Z", info["id_key"]], table, where=where
    )

    if info["has_time"] and get_timeseries:
        time, timeseries = read_timeseries(cur, table, info, where=where)
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
        }
    else:
        output = {
            "x": np.array(xx),
            "y": np.array(yy),
            "z": np.array(zz),
            "ps_id": np.array(ps_id),
            "mean_vel": np.array(mean_vel),
            "var_mean_vel": np.array(var_mean_vel),
        }

    return output


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
                f"SELECT {column} from {table} WHERE {where}"
            )
        ]
    else:
        result = [
            value[0] for value in cur.execute(f"SELECT {column} from {table}")
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
        for value in cur.execute(f"SELECT {cols} from {table} WHERE {where}"):
            for ii, val in enumerate(value):
                result[ii].append(val)
    else:
        for value in cur.execute(f"SELECT {cols} from {table}"):
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
        info["non_time_keys"] = [
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
        info["id_key"] = "ID"

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
    time = np.array([sql_key_to_time(k) for k in time_columns])
    cols = ""
    for cc in time_columns:
        cols += cc + ","
    cols = cols[:-1]
    if where:
        timeseries = np.array(
            cur.execute(f"SELECT {cols} from {table} WHERE {where}").fetchall()
        )
    else:
        timeseries = np.array(
            cur.execute(f"SELECT {cols} from {table}").fetchall()
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
    time = np.array([sql_key_to_time(k) for k in key_list])
    if where:
        queries = [
            (file_path, f"SELECT {k} from {table} WHERE {where}", jj)
            for jj, k in enumerate(key_list)
        ]
    else:
        queries = [
            (file_path, f"SELECT {k} from {table}", jj)
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


def read_buf(
    places_path: os.PathLike, shp_cfg: dict, kk: str, pool=False
) -> list:
    """Reads geometries from a gpkg file and returns them in buffered form

    :param places_path: The path to the places folder
    :type places_path: os.PathLike
    :param shp_cfg: The configuration for shape files
    :type shp_cfg: dict
    :param kk: The key of the fclass
    :type kk: str
    :return: The buffered shapes
    :rtype: list
    """
    geometries = load_osm_gpkg(
        os.path.join(places_path, shp_cfg["shp_file"][kk]),
        fclass=shp_cfg["fclass"][kk],
        pool=pool,
    )
    limit = 3 * 10**5
    if len(geometries) < limit and not pool:
        logging.info(f"Less than {limit} entries, non-parallel is faster.")
        buffered = [
            shapely.buffer(geom, shp_cfg["buffer_dist"][kk])
            for geom in geometries
        ]
    else:
        if not pool:
            logging.info("Starting parallel pool")
            with Pool(u4config.cpu_count) as p:
                buffered = p.starmap(
                    shapely.buffer,
                    zip(
                        geometries,
                        itertools.repeat(shp_cfg["buffer_dist"][kk]),
                    ),
                )
        else:
            logging.info("Using existing pool")
            buffered = pool.starmap(
                shapely.buffer,
                zip(
                    geometries,
                    itertools.repeat(shp_cfg["buffer_dist"][kk]),
                ),
            )

    return buffered


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
    else:
        raise KeyError("No keys for mean velocity found.")


def table_key_exists(fpath: os.PathLike, table_name: str, key: str) -> bool:
    """Checks if a particular `key` exists in the table in the sql database.

    :param fpath: The filepath to the database.
    :type fpath: os.PathLike
    :param table_name: The name of the table to che
    :type table_name: str
    :param key: The key to look for.
    :type key: str
    :return: True if the key exists in the table.
    :rtype: bool
    """

    con = sqlite3.connect(fpath)
    cur = con.cursor()
    info = read_info(cur, table_name)
    con.close()
    return key in info["all_keys"]


def add_new_key_value_pairs(
    fpath: os.PathLike, table_name: str, key: str, values: Iterable
):
    """Adds a new key and values to the table in the sql file.

    :param fpath: The path to the database.
    :type fpath: os.PathLike
    :param table_name: The name of the table.
    :type table_name: str
    :param key: The name of the key.
    :type key: str
    :param values: The values, size must be compatible with other entries!
    :type values: Iterable
    """

    with ogr.Open(fpath, update=1) as con:
        if table_key_exists(fpath, table_name=table_name, key=key):
            con.ExecuteSQL(f"ALTER TABLE {table_name} DROP COLUMN {key}")
        con.ExecuteSQL(
            f"ALTER TABLE {table_name} ADD COLUMN {key} float(32)",
            dialect="OGRSQL",
        )
        values = [np.round(val, 2) for val in values]
        values_str = repr(values).replace("[", "(")
        values_str = values_str.replace("]", ")")
        logging.info("Writing values to sql table.")
        con.ExecuteSQL(
            f"UPDATE {table_name} SET ({key})={values_str}",
            dialect="SQLITE",
        )
