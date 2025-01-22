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
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.sql as u4sql
import u4py.utils.cmd_args as u4args

# import u4py.utils.config as u4config
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
    # # All points
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
            x_rng = np.arange(minx, maxx, raster_size)
            miny = np.floor(bounds.miny[0] / raster_size) * raster_size
            maxy = np.ceil(bounds.maxy[0] / raster_size) * raster_size
            y_rng = np.arange(miny, maxy, raster_size)

            # Get indices where to put data for each point
            ii_x = [
                np.argwhere(x_rng >= x).squeeze()[0]
                for x in tqdm(
                    merged_data["vertikal"]["x"],
                    desc="Getting x indices",
                    leave=False,
                )
            ]
            ii_y = [
                np.argwhere(y_rng >= y).squeeze()[0]
                for y in tqdm(
                    merged_data["vertikal"]["y"],
                    desc="Getting y indices",
                    leave=False,
                )
            ]
            indices = np.array([(ix, iy) for ix, iy in zip(ii_x, ii_y)])
            uqidc = np.unique(indices, axis=0)
            extracts = [
                {
                    "x": x_rng[ii_x[ii]],
                    "y": y_rng[ii_y[ii]],
                    "time": merged_data["vertikal"]["time"],
                    "ps_id": [],
                    "timeseries_v": [],
                    "timeseries_ew": [],
                }
                for ii in tqdm(
                    range(len(ii_x)),
                    desc="Preallocating extracts",
                    leave=False,
                )
            ]
            for ii, (tsv, tsew, psid) in tqdm(
                enumerate(
                    zip(
                        merged_data["vertikal"]["timeseries"],
                        merged_data["Ost_West"]["timeseries"],
                        merged_data["vertikal"]["ps_id"],
                    )
                ),
                desc="Getting extracts",
                leave=False,
                total=len(merged_data["vertikal"]["x"]),
            ):
                uqiis = np.nonzero(np.all(uqidc == indices[ii], axis=1))[0]
                if len(uqiis) > 1:
                    logging.info("Error, to many indices found?")
                else:
                    extracts[uqiis[0]]["timeseries_v"].append(tsv)
                    extracts[uqiis[0]]["timeseries_ew"].append(tsew)
                    extracts[uqiis[0]]["ps_id"].append(psid)
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
