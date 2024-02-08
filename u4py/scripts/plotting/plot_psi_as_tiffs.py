"""
Plots all PSI from the gpkg file into geotiffs and saves them in a folder.
"""

import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        required=["psi_path", "output_path"],
        # interactive=False,
    )
    direction = "vertikal"

    # Setting up paths
    output_dir = os.path.join(
        project["paths"]["output_path"], f"PSI_Tiffs_{direction}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Reading common data, elevation data (zs) not used atm
    xs, ys, _, time, queries, info = u4sql.gen_queries_psi_gpkg(
        os.path.join(project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"),
        direction,
    )
    xq = np.array((xs - np.min(xs)) / 50, dtype=int)
    yq = np.array((ys - np.min(ys)) / 50, dtype=int)
    extent = (  # Adjustments necessary to match up coordinates
        np.min(xs) + 25,
        np.max(xs) + 75,
        np.min(ys) - 25,
        np.max(ys) + 25,
    )

    # Build arguments list (for easier parallelization, WIP)
    arguments = [(xq, yq, time, extent, output_dir, info, q) for q in queries]
    for arg in tqdm(
        arguments,
        desc="Generating GeoTiffs",
        leave=False,
    ):
        multi_geotiff(arg)


def multi_geotiff(arguments: tuple):
    """Takes the values in the tuple and creates a geotiff.

    The arguments contain data that is the same for all geotiffs and a sql
    query that is used to read the data from the input file.

    :param arguments: The arguments for plotting
    :type arguments: tuple
    """
    # Disassemble arguments
    xq, yq, time, extent, output_dir, info, query = arguments

    # Read data and fit them into velocity array
    data = u4sql.single_query(*query)
    vels = np.ones((np.max(yq) + 1, np.max(xq) + 1)) * np.nan
    vel = np.array([d if d else np.nan for d in data[0]])
    vels[yq, xq] = vel

    # Read timestamp
    time_str = datetime.strftime(time[query[2]], "%Y-%m-%d")
    fname = f"{time_str}.tif"

    # Create GeoTiff
    u4tiff.ndarray_to_geotiff(
        vels,
        extent,
        os.path.join(output_dir, fname),
        crs=info["proj_crs"],
        time=time_str,
    )


if __name__ == "__main__":
    main()
