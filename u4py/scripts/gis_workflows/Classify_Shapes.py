"""
Takes any kind of polygon shape as input and extracts various information from
external datasets and online repositories for the specific region.

Can be the output of ADAfinder, GroundMotionAnalyzer or U4Py-Full Workflows.
"""

import logging
import os

import geopandas as gp
import numpy as np
from tqdm import tqdm

import u4py.addons.web_services as u4web
import u4py.analysis.classify as u4class
import u4py.analysis.processing as u4proc
import u4py.utils.cmd_args as u4args
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    print("Starting setup of classification.")
    args = u4args.load()
    if args.input:
        proj_path = args.input
    else:
        proj_path = (
            "/home/rudolf/Documents/umwelt4/Classify_ShapesHLNUG.u4project"
        )

    project = u4proj.get_project(
        proj_path=proj_path,
        required=[
            "base_path",
            "psi_path",
            "processing_path",
            "places_path",
            "diff_plan_path",
            "sites_path",
        ],
        interactive=False,
    )
    shp_cfg = u4config.get_shape_config()

    # Setting up paths
    shp_file = os.path.join(
        project["paths"]["sites_path"],
        "thresholded_contours_all_shapes.gpkg",
    )

    if not (os.path.exists(shp_file)):
        shp_file = os.path.join(
            project["paths"]["sites_path"], "RD_Rutschungen_gesamt.shp"
        )

    # Getting Data
    shp_gdf = gp.read_file(shp_file).to_crs("EPSG:32632")
    if "groups" in shp_gdf.keys():
        unique_groups = np.unique(shp_gdf.groups)
    else:
        logging.info(
            "The dataset is the HLNUG Landslide database, extracting only the Landslides"
        )
        shp_gdf = shp_gdf[shp_gdf.OBJEKT == "Rutschung"]
        shp_gdf["groups"] = shp_gdf.AMT_NR_
        unique_groups = np.unique(shp_gdf.groups)
    if project.getboolean("config", "use_online"):
        u4web.set_client()

    kwargs = [
        {
            "shp_gdf": shp_gdf,
            "group": group,
            "buffer_size": 0.1,
            "shp_cfg": shp_cfg,
            "project": project,
            "use_online": project.getboolean("config", "use_online"),
            "use_internal": project.getboolean("config", "use_internal"),
            "save_report": True,
        }
        for group in unique_groups  # [:50]
    ]
    print("Starting Classification.")
    if project.getboolean("config", "use_parallel"):
        main_list = u4proc.batch_mapping(
            kwargs, classifier_wrapper, "Classifying Groups"
        )
    else:
        main_list = []
        for kwarg in tqdm(
            kwargs,
            total=len(kwargs),
            desc="Classifying Groups",
            leave=False,
        ):
            main_list.append(classifier_wrapper(kwarg))
    main_results = u4class.preallocate_results()
    for kk in main_results.keys():
        main_results[kk] = []

    print("Finished Queries, staring reformatting.")

    for res in tqdm(main_list, desc="Reformatting Results List"):
        if res:
            for kk in res.keys():
                if isinstance(res[kk], gp.GeoDataFrame):
                    main_results[kk].append("[]")
                elif isinstance(res[kk], list):
                    if len(res[kk]) > 1:
                        main_results[kk].append(str(res[kk]))
                    elif len(res[kk]) == 1:
                        main_results[kk].append(str(res[kk][0]))
                    else:
                        main_results[kk].append(str(res[kk]))
                else:
                    main_results[kk].append(res[kk])
    print("Creating final dataframe.")
    main_gdf = gp.GeoDataFrame(data=main_results, crs=shp_gdf.crs)
    print("Saving final dataframe")
    main_gdf.to_file(
        os.path.join(project["paths"]["sites_path"], "Classified_Shapes.gpkg")
    )


def classifier_wrapper(kwargs: dict) -> dict:
    """Parallel processing wrapper for classify_shape()

    :param args: The arguments
    :type args: dict
    :return: The results dictionary.
    :rtype: dict
    """
    res = u4class.classify_shape(**kwargs)
    return res


if __name__ == "__main__":
    main()
