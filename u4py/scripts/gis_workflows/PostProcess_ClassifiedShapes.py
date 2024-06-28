"""
Uses the shapes classified by the main workflow and generates a report for
each anomaly.
"""

import configparser
import datetime
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import geopandas as gp
import numpy as np
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.io.reporting as u4rep
import u4py.plotting.plots as u4plots
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

warnings.filterwarnings("ignore")
# u4config.cpu_count = 60


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/PostProcess_ClassifiedShapes.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "places_path",
            "output_path",
            "results_path",
            "diff_plan_path",
        ],
        interactive=False,
    )
    overwrite = True
    use_parallel = True
    generate_plots = True
    generate_pdf = True

    # Setting up paths
    output_path = os.path.join(
        project["paths"]["output_path"], "Detailed_Maps"
    )
    os.makedirs(output_path, exist_ok=True)
    class_shp_fp = os.path.join(
        project["paths"]["results_path"], "Classified_Shapes.gpkg"
    )
    cls_shp_fp_filtered = os.path.join(
        project["paths"]["results_path"], "Filtered_Classified_Shapes.gpkg"
    )
    hlnug_path = os.path.join(
        project["paths"]["places_path"],
        "Classifier_shapes",
        "classifier_shapes.gpkg",
    )
    dem_path = os.path.join(
        os.path.split(project["paths"]["diff_plan_path"])[0],
        "DGM1_2021",
        "raster_2021",
    )
    contour_path = os.path.join(
        project["paths"]["results_path"],
        "thresholded_contours_all_shapes.gpkg",
    )

    # Read Data
    if not os.path.exists(cls_shp_fp_filtered) or overwrite:
        gdf_filtered = filter_shapes(
            class_shp_fp, cls_shp_fp_filtered, project
        )
    else:
        gdf_filtered = gp.read_file(cls_shp_fp_filtered)

    # Generating Plots
    if generate_plots:
        # Assembling arguments for processing
        args = [
            (
                row,
                gdf_filtered.crs,
                output_path,
                hlnug_path,
                project,
                dem_path,
                contour_path,
            )
            for row in gdf_filtered.iterrows()
        ]
        if use_parallel:
            with Pool(u4config.cpu_count - 2) as p:
                list(
                    tqdm(
                        p.imap_unordered(wrap_map_worker, args),
                        total=len(args),
                        desc="Generating Plots",
                        leave=False,
                    )
                )
        else:
            for arg in tqdm(
                args,
                total=len(args),
                desc="Generating Plots",
                leave=False,
            ):
                wrap_map_worker(arg)

    # Generating TeX and final PDF
    if generate_pdf:
        for row in tqdm(
            gdf_filtered.iterrows(),
            desc="Generating tex files",
            total=len(gdf_filtered),
        ):
            u4rep.site_report(row, output_path, "tex_includes")
        u4rep.main_report(output_path)


def filter_shapes(
    input_path: os.PathLike,
    output_path: os.PathLike,
    project: configparser.ConfigParser,
):
    """Filters the classified shapes by known geohazards

    :param input_path: The path to the file where all classified shapes are located.
    :type input_path: os.PathLike
    :param output_path: Path where to save the filtered shapes as shapefile.
    :type output_path: os.PathLike
    :param project: The project config for paths etc...
    :type project: configparser.ConfigParser
    """
    class_shp_gdf = gp.read_file(input_path)

    lands_f = class_shp_gdf.landslides_num_inside > 0
    karst_f = class_shp_gdf.karst_num_inside > 0
    rockf_f = class_shp_gdf.rockfall_num_inside > 0
    filter_all = np.logical_or(np.logical_or(lands_f, karst_f), rockf_f)
    gdf_filtered = class_shp_gdf[filter_all]

    # Do a reverse geocoding to get the address of the locations.
    cached_locations = os.path.join(
        project["paths"]["results_path"], "cached_locations.txt"
    )
    if not os.path.exists(cached_locations):
        locations = u4spatial.reverse_geolocate(gdf_filtered)
        with open(cached_locations, "wt", encoding="utf-8") as cache:
            for loc in locations:
                cache.write(loc + "\n")
    else:
        with open(
            cached_locations, "rt", encoding="utf-8", newline="\n"
        ) as cache:
            locations = cache.readlines()
    if len(locations) != len(gdf_filtered):
        locations = u4spatial.reverse_geolocate(gdf_filtered)
        with open(cached_locations, "wt", encoding="utf-8") as cache:
            for loc in locations:
                cache.write(loc + "\n")
    gdf_filtered = gdf_filtered.assign(locations=locations)

    gdf_filtered.to_file(output_path)
    gdf_filtered.to_crs("EPSG:4326").to_file(
        os.path.splitext(output_path)[0] + ".geojson"
    )
    return gdf_filtered


def wrap_map_worker(args: tuple):
    """Parallel wrapper for `map_worker()`

    :param args: The function arguments.
    :type args: tuple
    """
    map_worker(*args)


def map_worker(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    hlnug_path: os.PathLike,
    project: configparser.ConfigParser,
    dem_path: os.PathLike,
    contour_path: os.PathLike,
):
    """Calls the various plotting and reporting functions.

    :param row: The index and entry of a row in a GeoDataFrame of classified shapes.
    :type row: tuple
    :param crs: The crs of the dataset.
    :type crs: str
    :param output_path: The path where the output shall be saved.
    :type output_path: os.PathLike
    :param hlnug_path: The path where HLNUG data is found.
    :type hlnug_path: os.PathLike
    :param project: The project's config.
    :type project: configparser.ConfigParser
    :param dem_path: The path to the digital elevation model.
    :type dem_path: os.PathLike
    :param contour_path: The path to the contour file.
    :type contour_path: os.PathLike
    """

    u4plots.detailed_map(
        row,
        crs,
        output_path,
        "known_features",
        hlnug_path,
        contour_path,
        plot_buffer=250,
    )
    u4plots.satimg_map(
        row, crs, output_path, "known_features", contour_path, plot_buffer=100
    )
    u4plots.diffplan_map(
        row,
        crs,
        output_path,
        "known_features",
        project["paths"]["diff_plan_path"],
        plot_buffer=100,
    )
    u4plots.dem_map(
        row,
        crs,
        output_path,
        "known_features",
        dem_path,
        contour_path,
        plot_buffer=100,
    )
    u4plots.slope_map(
        row,
        crs,
        output_path,
        "known_features",
        dem_path,
        contour_path,
        plot_buffer=100,
    )
    u4plots.aspect_map(
        row,
        crs,
        output_path,
        "known_features",
        dem_path,
        contour_path,
        plot_buffer=100,
    )
    u4plots.aspect_slope_map(
        row,
        crs,
        output_path,
        "known_features",
        dem_path,
        contour_path,
        plot_buffer=100,
    )
    u4plots.geology_map(
        row,
        crs,
        output_path,
        "known_features",
        contour_path,
        os.path.join(
            project["paths"]["places_path"], "Geologie (Kartiereinheiten).pkl"
        ),
        plot_buffer=100,
    )
    u4plots.hydrogeology_map(
        row,
        crs,
        output_path,
        "known_features",
        contour_path,
        os.path.join(
            project["paths"]["places_path"], "Hydrogeologische Einheiten.pkl"
        ),
        plot_buffer=100,
    )
    u4plots.topsoil_map(
        row,
        crs,
        output_path,
        "known_features",
        contour_path,
        os.path.join(project["paths"]["places_path"], "legend_BFD50.pkl"),
        plot_buffer=100,
    )
    if row[1].timeseries_num_psi > 5:
        u4plots.timeseries_map(
            row,
            crs,
            output_path,
            "known_features",
            contour_path,
            os.path.join(project["paths"]["psi_path"]),
            plot_buffer=100,
        )


if __name__ == "__main__":
    tic = datetime.datetime.now()
    main()
    duration = datetime.datetime.now() - tic
    print(f"Runtime: {duration}")
