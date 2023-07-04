"""
Contains some sqlite functions for working with gpkg files
"""
import logging
import os
import sqlite3 as sql
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Tuple

import numpy as np
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
    con = sql.connect(file_path)
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
    con = sql.connect(file_path)
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
    con = sql.connect(file_path)
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
