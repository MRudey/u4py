"""
Plots the results from inversion processing that has been saved as a pickle
file.

Additionally saves both components as geotiffs.
"""

import os

import u4py.plotting.plots as u4plots
import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        required=[
            "results_path",
            "base_path",
            "base_map_path",
            "tektonik_path",
        ]
    )

    data = u4files.get_pickled_inversion_results(
        project["paths"]["results_path"]
    )
    converted_data = u4plotprep.convert_results_for_grid(data[0])
    lin_2d, sin_2d, extend = u4plotprep.make_gridded_data(*converted_data)

    u4files.ndarray_to_geotiff(
        lin_2d,
        extend,
        os.path.join(
            project["paths"]["base_path"],
            "INSAR_plots",
            "linear_trend_2023.tif",
        ),
        crs="EPSG:32632",
    )
    u4files.ndarray_to_geotiff(
        sin_2d,
        extend,
        os.path.join(
            project["paths"]["base_path"],
            "INSAR_plots",
            "seasonal_trend_2023.tif",
        ),
        crs="EPSG:32632",
    )
    u4plots.plot_gridded(
        lin_2d,
        sin_2d,
        extend,
        tektonik_path=project["paths"]["tektonik_path"],
        base_map_path=project["paths"]["base_map_path"],
        dpi=100,
    )


if __name__ == "__main__":
    main()
