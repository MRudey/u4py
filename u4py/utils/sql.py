""" Contains all sqlite functions for working with gpkg files """
import logging
import os
import sqlite3 as sql
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


def get_table_names(file_path: os.PathLike) -> list:
    con = sql.connect(file_path)
    cur = con.cursor()

    # Get all table names
    tables = [
        res[0]
        for res in cur.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name LIKE 'Zeitreihe_%'"
        )
    ]
    con.close()
    return tables


def map_queries(queries):
    """Maps Queries to a parallel processing pool"""
    logging.info(f"Starting parallel sql extraction.")
    with Pool() as p:
        results = list(
            tqdm(
                p.map(multi_proc_query, queries),
                total=len(queries),
                desc="Processing queries",
                leave=False,
            )
        )
    return results


def multi_proc_query(args):
    """Multiprocessing wrapper for sql queries"""
    return single_query(*args)


def single_query(file_path: os.PathLike, query: str, jj: int = -1):
    """Executes a single sql query for the given db-file"""
    logging.debug(f"{query}")
    con = sql.connect(file_path)
    cur = con.cursor()
    result = [value[0] for value in cur.execute(query)]
    con.close()
    if jj >= 0:
        return (result, jj)
    else:
        return result


def sql_key_to_time(key):
    """Returns a datetime object for the given date"""
    return datetime.strptime(key, "date_%Y%m%d")


def table_to_dict(file_path: os.PathLike, table: str) -> dict:
    """
    Opens the given sql database and gets all content of the given table
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
    elif "PS_ID" in all_keys:
        non_time_keys = ["X", "Y", "Z", "PS_ID", "Shape", "OBJECTID"]
        id_key = "PS_ID"
    if "Input" in all_keys:
        non_time_keys = [
            "X",
            "Y",
            "Z",
            "ID",
            "Input",
            "mean_velo_city",
            "var_mean_velocity",
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
        logging.info(f"Getting timeseries")
        key_list = [k for k in all_keys if k not in non_time_keys]
        time = np.array([sql_key_to_time(k) for k in key_list])
        num_fields = len(key_list)
        timeseries = np.zeros((num_points, num_fields))

        queries = [
            (file_path, f"SELECT {k} from {table}", jj)
            for jj, k in enumerate(key_list)
        ]
        results = map_queries(queries)
        for r in results:
            timeseries[:, r[1]] = np.array(r[0])
    else:
        logging.info(f"Getting means.")
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
