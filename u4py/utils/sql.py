"""
Contains some sqlite functions for working with gpkg files
"""
from __future__ import annotations

import itertools
import logging
import os
import sqlite3
import struct
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Tuple

import numpy as np
import shapely
import utm
from tqdm import tqdm

import u4py.utils.config as u4config


def get_table_names(file_path: os.PathLike) -> list:
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


def table_to_dict(file_path: os.PathLike, table: str) -> dict:
    """Opens the given sql database and gets all content of the given table.

    :param file_path: The path to the database.
    :type file_path: os.PathLike
    :param table: The Table to get from.
    :type table: str
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
    all_keys = [res[1] for res in cur.execute(f"PRAGMA TABLE_INFO({table})")]

    # Define type of file:
    has_time = True
    if "stack_ID" in all_keys:
        has_time = False
        mean_vel = np.zeros(num_points)
        var_mean_vel = np.zeros(num_points)
    elif "PS_ID" in all_keys:  # ASCE and DESC Data
        non_time_keys = ["X", "Y", "Z", "PS_ID", "Shape", "OBJECTID"]
        id_key = "PS_ID"
    if "Input" in all_keys:  # L3 Data
        non_time_keys = [
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
        id_key = "ID"

    # Get Coordinates
    xx = np.array(
        [value[0] for value in cur.execute(f"SELECT X from {table}")]
    )
    yy = np.array(
        [value[0] for value in cur.execute(f"SELECT Y from {table}")]
    )
    zz = np.array(
        [value[0] for value in cur.execute(f"SELECT Z from {table}")]
    )
    ps_id = np.array(
        [value[0] for value in cur.execute(f"SELECT {id_key} from {table}")]
    )

    if has_time:
        con.close()
        logging.debug(f"Getting timeseries")
        key_list = [k for k in all_keys if k not in non_time_keys]
        time = np.array([sql_key_to_time(k) for k in key_list])
        num_fields = len(key_list)
        timeseries = np.zeros((num_points, num_fields))

        queries = [
            (file_path, f"SELECT {k} from {table}", jj)
            for jj, k in enumerate(key_list)
        ]
        results = map_queries(queries)
        logging.debug("Aggregating results of queries to timeseries array.")
        for r in results:
            timeseries[:, r[1]] = np.array(r[0])
    else:
        logging.debug(f"Getting means.")
        mean_vel = np.array(
            [
                value[0]
                for value in cur.execute(f"SELECT mean_velocity from {table}")
            ]
        )
        var_mean_vel = np.array(
            [
                value[0]
                for value in cur.execute(
                    f"SELECT var_mean_velocity from {table}"
                )
            ]
        )
        con.close()

    if has_time:
        output = {
            "x": xx,
            "y": yy,
            "z": zz,
            "time": time,
            "ps_id": ps_id,
            "timeseries": timeseries,
        }
    else:
        output = {
            "x": xx,
            "y": yy,
            "z": zz,
            "ps_id": ps_id,
            "mean_vel": mean_vel,
            "var_mean_vel": var_mean_vel,
        }

    return output


def load_tables(file_path: os.PathLike) -> dict:
    """Loads content of all tables in the given sql database and returns as a data dictionary.

    :param file_path: The input database.
    :type file_path: os.PathLike
    :return: The output dictionary.
    :rtype: dict
    """
    data = dict()
    tables = get_table_names(file_path)
    for table in tqdm(tables, desc="Reading from tables"):
        data[table] = table_to_dict(file_path, table)
    return data


def load_gpkg(gpkg_file: os.PathLike, fclass: list = [], pool=False):
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
    :rtype: ogr.Geometry

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
    geometries = load_gpkg(
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
