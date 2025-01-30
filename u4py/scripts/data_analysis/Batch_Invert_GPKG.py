"""
Inverts all timeseries in a GPKG file and outputs the results as a pkl file.
"""

import configparser
import logging
import os
import pickle
from pathlib import Path

import geopandas as gp
import numpy as np
import scipy.spatial as spspat
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.sql as u4sql
import u4py.utils.cmd_args as u4args
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4proj

# u4config.start_logger()


def main():
    args = u4args.load()
    if args.input:
        proj_path = args.input
    else:
        proj_path = Path(
            "~/Documents/umwelt4/BatchInvert_EGMS_2015-2021.u4project"
        ).expanduser()
    overwrite = args.overwrite
    overwrite = True

    project = u4proj.get_project(
        required=[
            "base_path",
            "psi_path",
            "psivert_path",
            "psiew_path",
            "output_path",
        ],
        proj_path=proj_path,
    )
    merged_data = merge_data(project)
    # All points
    # if not os.path.exists(project["paths"]["output_path"]) or overwrite:
    #     extracts = u4proc.get_extracts(merged_data)
    #     invert_extracts(extracts, project["paths"]["output_path"])

    # Gridded points
    logging.info("Starting gridding analysis")
    raster_sizes = [250, 1000, 2500]
    bounds = u4sql.get_bounds(project["paths"]["psivert_path"])
    crs = u4sql.get_crs(project["paths"]["psivert_path"])
    bounds = (
        gp.GeoDataFrame(
            geometry=[u4spatial.bounds_to_polygon(bounds[0])], crs=crs[0]
        )
        .to_crs("EPSG:3035")
        .bounds
    )

    for raster_size in raster_sizes:
        output_path = os.path.join(
            project["paths"]["psi_path"] + f"_{raster_size}m_inv_results.pkl"
        )
        if not os.path.exists(output_path) or overwrite:
            logging.info(f"Subdivision size {raster_size}")

            # Get x and y extend for rastering
            minx = np.floor(bounds.minx[0] / raster_size) * raster_size
            maxx = np.ceil(bounds.maxx[0] / raster_size) * raster_size
            miny = np.floor(bounds.miny[0] / raster_size) * raster_size
            maxy = np.ceil(bounds.maxy[0] / raster_size) * raster_size

            # Build KDTree and queries for fast lookup
            x, y = np.mgrid[minx:maxx:raster_size, miny:maxy:raster_size]
            tree = spspat.KDTree(np.c_[x.ravel(), y.ravel()])
            queries = [
                (xx, yy)
                for xx, yy in zip(
                    merged_data["vertikal"]["x"], merged_data["vertikal"]["y"]
                )
            ]

            # Lookup indices of grid where to put the data
            dd, idx = tree.query(queries, workers=u4config.cpu_count)

            logging.info("Preallocating lists")
            ts_v = [list() for _ in range(x.size)]
            ts_ew = [list() for _ in range(x.size)]
            ps_id = [list() for _ in range(x.size)]
            time = [list() for _ in range(x.size)]
            xs = [0 for _ in range(x.size)]
            ys = [0 for _ in range(x.size)]
            x_flat = np.array(x).flat
            y_flat = np.array(y).flat

            # Assemble data lists
            for ii, jj in tqdm(
                enumerate(idx), total=len(idx), desc="Assemble data lists"
            ):
                ts_v[jj].append(merged_data["vertikal"]["timeseries"][ii])
                ts_ew[jj].append(merged_data["Ost_West"]["timeseries"][ii])
                ps_id[jj].append(merged_data["vertikal"]["ps_id"][ii])
                time[jj].append(merged_data["vertikal"]["time"])
                xs[jj] = x_flat[jj]
                ys[jj] = y_flat[jj]

            # Extracting data
            extracts = []
            for ii in tqdm(
                range(len(ts_v)), total=len(ts_v), desc="Extracting data"
            ):
                if len(ts_v[ii]) > 0:
                    extracts.append(
                        u4convert.reformat_gpkg(
                            xs[ii],
                            ys[ii],
                            0,
                            ps_id[ii],
                            np.hstack(time[ii]),
                            np.hstack(ts_v[ii]),
                            np.hstack(ts_ew[ii]),
                        )
                    )
            invert_extracts(extracts, output_path)


def merge_data(project: configparser.ConfigParser) -> dict:
    # Set Paths
    table_v = os.path.splitext(
        os.path.split(project["paths"]["psivert_path"])[-1]
    )[0]
    table_ew = os.path.splitext(
        os.path.split(project["paths"]["psiew_path"])[-1]
    )[0]
    # Load Data
    logging.info("Loading vertical data")
    data_v = u4sql.table_to_dict(project["paths"]["psivert_path"], table_v)
    logging.info("Loading east-west data")
    data_ew = u4sql.table_to_dict(project["paths"]["psiew_path"], table_ew)

    logging.info("Creating sort indices")
    sidx_v = np.argsort(data_v["ps_id"])
    sidx_ew = np.argsort(data_ew["ps_id"])
    logging.info("Sorting data")
    to_sort = ["x", "y", "z", "ps_id", "timeseries"]
    for kk in to_sort:
        data_v[kk] = data_v[kk][sidx_v]
        data_ew[kk] = data_ew[kk][sidx_ew]

    return {"vertikal": data_v, "Ost_West": data_ew}


def invert_extracts(extracts: list[dict], output_path: os.PathLike):
    results = u4proc.batch_mapping(
        extracts, u4proc.inversion_map_worker, "Inverting Data"
    )

    with open(output_path, "wb") as pkl_file:
        pickle.dump(results, pkl_file)


if __name__ == "__main__":
    main()
