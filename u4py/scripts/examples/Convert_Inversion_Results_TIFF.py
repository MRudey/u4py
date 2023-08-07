"""
Converts the pickled inversion results to a geotiff.
"""

import os

import numpy as np

import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        required=[
            "results_path",
            "output_path",
        ],
        interactive=False,
    )
    is_normal = True
    _, fname = os.path.split(project["paths"]["results_path"])
    fname, _ = os.path.splitext(fname)

    if is_normal:
        data = u4files.get_pickled_inversion_results(
            project["paths"]["results_path"]
        )
        converted_data, chunk_size = u4plotprep.convert_results_for_grid(
            data[0], chunk_size=2500
        )
    else:
        data = u4files.get_all_pickle_data(project["paths"]["results_path"])
        converted_data, chunk_size = u4plotprep.convert_results_for_grid(
            data[1], chunk_size=50
        )
    grids, extend = u4plotprep.make_gridded_data(converted_data, chunk_size)
    grid_names = [
        "linear_trend",
        "annual_sine",
        "annual_cosine",
        "semiannual_sine",
        "semiannual_cosine",
    ]

    output_dir = os.path.join(
        project["paths"]["output_path"], fname + "_tiffs"
    )
    os.makedirs(output_dir, exist_ok=True)
    for grid, name in zip(grids, grid_names):
        u4files.ndarray_to_geotiff(
            grid,
            extend,
            os.path.join(
                output_dir,
                f"{name}_{chunk_size}_{fname}.tif",
            ),
            crs="EPSG:32632",
        )

    for grid, name in zip(grids, grid_names):
        u4files.ndarray_to_geotiff(
            np.abs(grid),
            extend,
            os.path.join(
                output_dir,
                f"abs_{name}_{chunk_size}_{fname}.tif",
            ),
            crs="EPSG:32632",
        )


if __name__ == "__main__":
    main()
