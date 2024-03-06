"""
Takes all areas that are classified as quarries or tailings and extracts LIDAR
differences.
"""

import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt

import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.utils.projects as u4proj


def main():
    overwrite = False
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Extract_Quarry_Data.u4project"
        ).expanduser(),
        required=["places_path", "diff_plan_path", "output_path"],
        # interactive=False,
    )
    osm_path = os.path.join(project["paths"]["places_path"], "all_shapes.gpkg")
    volumes_path = os.path.join(
        project["paths"]["output_path"], "quarry_landfill_diff_volumes.shp"
    )

    if not os.path.exists(volumes_path) or overwrite:

        quarry_shapes = u4sql.load_osm_gpkg(
            osm_path,
            fclass=["quarry"],
            table_name="gis_osm_landuse_a_free_1",
        )
        landfill_gdf = gp.read_file(
            os.path.join(project["paths"]["places_path"], "SHP", "lan-lan.shp")
        )
        merged_geometries = []
        labels = ["quarry" for ii in quarry_shapes]
        labels.extend(["landfill" for ii in range(len(landfill_gdf))])
        merged_geometries.extend(quarry_shapes)
        merged_geometries.extend(landfill_gdf.geometry.to_list())

        merged_gdf = gp.GeoDataFrame(
            geometry=merged_geometries,
            data={"labels": labels},
            crs="EPSG:32632",
        )
        volumes = u4tiff.calculate_volume_in_shape(
            merged_gdf, project["paths"]["diff_plan_path"]
        )
        volumes.to_file(volumes_path)
    else:
        volumes = gp.read_file(volumes_path)

    fig, ax = plt.subplots()
    volumes.plot(ax=ax, column="volume")
    plt.show()


if __name__ == "__main__":
    main()
