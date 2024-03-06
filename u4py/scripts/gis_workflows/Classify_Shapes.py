"""
Takes any kind of polygon shape as input and extracts various information from
external datasets and online repositories for the specific region.

Can be the output of ADAfinder, GroundMotionAnalyzer or U4Py-Full Workflows.
"""

import logging
import os
from pathlib import Path

import geopandas as gp
import numpy as np
import shapely as shp
from tqdm import tqdm

import u4py.analysis.classify as u4class
import u4py.io.gpkg as u4gpkg
import u4py.plotting.plots as u4plots
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Classify_Shapes.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "psi_path",
            "processing_path",
            "places_path",
            "diff_plan_path",
            "output_path",
        ],
        interactive=False,
    )
    shp_cfg = u4config.get_shape_config()

    # Setting up paths
    shp_file = os.path.join(
        project["paths"]["processing_path"],
        "thresholded_contours_all_shapes.gpkg",
    )
    site_stat_path = os.path.join(project["paths"]["output_path"], "SiteStats")
    os.makedirs(site_stat_path, exist_ok=True)

    # Getting Data
    sub_region = create_test_region()
    shp_gdf = u4gpkg.load_gpkg_data_region_ogr(sub_region, shp_file)

    unique_groups = np.unique(shp_gdf.groups)
    for group in tqdm(unique_groups, desc="Classifying Groups"):
        res = u4class.classify_shape(
            shp_gdf=shp_gdf,
            group=group,
            buffer_size=100,
            shp_cfg=shp_cfg,
            project=project,
            use_online=True,
            save_report=True,
        )
        # if res:
        #     print(res)
        #     u4plots.plot_site_statistics(res, site_stat_path)


def create_test_region():
    """
    Gets example region to clip dataset.
    """
    test_region = gp.GeoDataFrame(
        geometry=[
            shp.Polygon(
                [
                    (553000, 5686000),
                    (573000, 5686000),
                    (573000, 5667000),
                    (553000, 5667000),
                    (553000, 5686000),
                ]
            )
        ],
        crs="EPSG:32632",
    )
    test_region.to_file(
        Path(r"~\Documents\ArcGIS\SelectedSites\test_region.shp").expanduser()
    )
    return test_region


if __name__ == "__main__":
    main()
