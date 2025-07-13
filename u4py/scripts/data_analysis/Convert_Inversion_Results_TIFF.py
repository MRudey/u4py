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
from pathlib import Path

import numpy as np
from tqdm import tqdm

import u4py.analysis.other as u4other
import u4py.io.files as u4files
import u4py.io.psi as u4psi
import u4py.io.tiff as u4tiff
import u4py.plotting.preparation as u4plotprep

# import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

# u4config.start_logger()


def main():
    project = u4proj.get_project(
        required=["results_path"],
        interactive=False,
        proj_path=Path(
            "~/Documents/umwelt4/Pkl_to_Tiff_EGMS.u4project"
        ).expanduser(),
    )

    logging.info("Setting up paths")
    # Getting list of all pkl files in folder
    fnames = [
        fp
        for fp in os.listdir(project["paths"]["results_path"])
        if fp.endswith(".pkl")
    ]
    chunk_sizes = []
    for fn in fnames:
        if "EGMS" in fn:
            is_egms = False
            if "vertikal_inversion_results" in fn:
                chunk_sizes.append(100)
            elif "5000m" in fn:
                chunk_sizes.append(5000)
            elif "2500m" in fn:
                chunk_sizes.append(2500)
            elif "250m" in fn:
                chunk_sizes.append(250)
            elif "1000m" in fn:
                chunk_sizes.append(1000)
            elif "500m" in fn:
                chunk_sizes.append(500)
        else:
            raise ValueError("Could not get chunksize from filename.")

    os.makedirs(project["paths"]["results_path"], exist_ok=True)
    fail_list = []
    for fname, chunk_size in tqdm(
        zip(fnames, chunk_sizes), desc="Converting files", total=len(fnames)
    ):
        failed = convert_pkl_to_tiff(
            fname, project["paths"]["results_path"], is_egms, chunk_size
        )
        if failed:
            fail_list.append(failed)
    if fail_list:
        print(" ### Script failed for: ### ")
        for fl in fail_list:
            print(fl)


def convert_pkl_to_tiff(
    full_name: str,
    output_dir: os.PathLike | str,
    is_normal: bool,
    chunk_size: int,
):
    pkl_path = os.path.join(output_dir, full_name)
    fname = full_name[full_name.index("EGMS") : full_name.index("EGMS") + 14]

    logging.info("Loading and rearranging data for further analysis")
    if is_normal:
        data = u4psi.get_pickled_inversion_results(pkl_path)
        if data[0] and data[1]:
            converted_data, chunk_size = u4plotprep.convert_results_for_grid(
                data[0], chunk_size=chunk_size
            )
        else:
            converted_data = []
    else:
        data = u4files.get_all_pickle_data(pkl_path)
        if data[0] and data[1]:
            converted_data, chunk_size = u4plotprep.convert_results_for_grid(
                data[1], chunk_size=chunk_size
            )
        else:
            converted_data = []

    if len(converted_data) > 0:
        create_grids(
            converted_data, chunk_size, output_dir, fname, crs="EPSG:3035"
        )
    else:
        return full_name


def create_grids(
    converted_data: list,
    chunk_size: int,
    output_dir: os.PathLike | str,
    fname: str,
    crs: str,
):
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
        u4tiff.ndarray_to_geotiff(
            grid,
            extend,
            os.path.join(
                output_dir,
                f"{name}_{chunk_size}_{fname}.tif",
            ),
            crs=crs,
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
        u4tiff.ndarray_to_geotiff(
            np.abs(grid),
            extend,
            os.path.join(
                output_dir,
                f"abs_{name}_{chunk_size}_{fname}.tif",
            ),
            crs=crs,
            compress="lzw",
        )

    logging.info("Maximum of annual")
    max_vals, max_time = u4other.find_maximum_sines(
        grids[1], grids[2], interval=365
    )
    u4tiff.ndarray_to_geotiff(
        max_vals,
        extend,
        os.path.join(
            output_dir,
            f"max_vals_annual_{chunk_size}_{fname}.tif",
        ),
        crs=crs,
        compress="lzw",
    )
    u4tiff.ndarray_to_geotiff(
        max_time,
        extend,
        os.path.join(
            output_dir,
            f"max_time_annual_{chunk_size}_{fname}.tif",
        ),
        crs=crs,
        compress="lzw",
    )

    logging.info("Maximum of semiannual")
    max_vals, max_time = u4other.find_maximum_sines(
        grids[3], grids[4], interval=365 / 2
    )
    u4tiff.ndarray_to_geotiff(
        max_vals,
        extend,
        os.path.join(
            output_dir,
            f"max_vals_semiannual_{chunk_size}_{fname}.tif",
        ),
        crs=crs,
        compress="lzw",
    )
    u4tiff.ndarray_to_geotiff(
        max_time,
        extend,
        os.path.join(
            output_dir,
            f"max_time_semiannual_{chunk_size}_{fname}.tif",
        ),
        crs=crs,
        compress="lzw",
    )


if __name__ == "__main__":
    main()
