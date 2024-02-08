"""
Converts the pickled inversion results to geotiffs.

Also computes some additional parameters from the inversion results. Exports
the following data as geotiffs:

    - Linear component
    - Annual sine and cosine
    - Semiannual sine and cosine
    - Absolute value of annual sine and cosine
    - Absolute value of semiannual sine and cosine
    - Maxima and time of maxima for annual and semiannual motions
"""

import logging
import os

import numpy as np
from tqdm import tqdm

import u4py.analysis.other as u4other
import u4py.plotting.preparation as u4plotprep
import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        required=[
            "results_path",
            "output_path",
        ],
        interactive=False,
    )

    logging.info("Setting up paths")
    is_normal = False
    _, fname = os.path.split(project["paths"]["results_path"])
    fname, _ = os.path.splitext(fname)
    output_dir = os.path.join(
        project["paths"]["output_path"], fname + "_tiffs"
    )
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Loading and rearranging data for further analysis")
    if is_normal:
        data = u4files.get_pickled_inversion_results(
            project["paths"]["results_path"]
        )
        converted_data, chunk_size = u4plotprep.convert_results_for_grid(
            data[0], chunk_size=50
        )
    else:
        data = u4files.get_all_pickle_data(project["paths"]["results_path"])
        converted_data, chunk_size = u4plotprep.convert_results_for_grid(
            data[1], chunk_size=50
        )

    logging.info("Creating numpy arrays from data")
    grids, extend = u4plotprep.make_gridded_data(converted_data, chunk_size)
    grid_names = [
        "linear_trend",
        "annual_sine",
        "annual_cosine",
        "semiannual_sine",
        "semiannual_cosine",
    ]

    logging.info("Creating GeoTiffs")

    logging.info("Normal Components")
    for grid, name in tqdm(
        zip(grids, grid_names),
        desc="Saving arrays",
        total=len(grid_names),
        leave=False,
    ):
        grid[np.abs(grid) > np.nanpercentile(np.abs(grid), 99.99)] = np.nan
        u4files.ndarray_to_geotiff(
            grid,
            extend,
            os.path.join(
                output_dir,
                f"{name}_{chunk_size}_{fname}.tif",
            ),
            crs="EPSG:32632",
            compress="lzw",
        )

    logging.info("Absolute Components")
    for grid, name in tqdm(
        zip(grids, grid_names),
        desc="Saving arrays",
        total=len(grid_names),
        leave=False,
    ):
        grid[np.abs(grid) > np.nanpercentile(np.abs(grid), 99.99)] = np.nan
        u4files.ndarray_to_geotiff(
            np.abs(grid),
            extend,
            os.path.join(
                output_dir,
                f"abs_{name}_{chunk_size}_{fname}.tif",
            ),
            crs="EPSG:32632",
            compress="lzw",
        )

    logging.info(f"Maximum of annual")
    max_vals, max_time = u4other.find_maximum_sines(
        grids[1], grids[2], interval=365
    )
    u4files.ndarray_to_geotiff(
        max_vals,
        extend,
        os.path.join(
            output_dir,
            f"max_vals_annual_{chunk_size}_{fname}.tif",
        ),
        crs="EPSG:32632",
        compress="lzw",
    )
    u4files.ndarray_to_geotiff(
        max_time,
        extend,
        os.path.join(
            output_dir,
            f"max_time_annual_{chunk_size}_{fname}.tif",
        ),
        crs="EPSG:32632",
        compress="lzw",
    )

    logging.info(f"Maximum of semiannual")
    max_vals, max_time = u4other.find_maximum_sines(
        grids[3], grids[4], interval=365 / 2
    )
    u4files.ndarray_to_geotiff(
        max_vals,
        extend,
        os.path.join(
            output_dir,
            f"max_vals_semiannual_{chunk_size}_{fname}.tif",
        ),
        crs="EPSG:32632",
        compress="lzw",
    )
    u4files.ndarray_to_geotiff(
        max_time,
        extend,
        os.path.join(
            output_dir,
            f"max_time_semiannual_{chunk_size}_{fname}.tif",
        ),
        crs="EPSG:32632",
        compress="lzw",
    )


if __name__ == "__main__":
    main()
