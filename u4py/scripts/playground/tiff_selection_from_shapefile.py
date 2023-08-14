"""
Visualizes tiff selection via shapefile
"""

import geopandas as gp

import u4py.utils.files as u4files
import u4py.utils.projects as u4projects

# import matplotlib.pyplot as plt


def main():
    project = u4projects.get_project(
        required=["subsubregions_path", "diff_plan_path"]
    )

    all_tiff_gdf = u4files.get_all_tiff_regions(project, overwrite=True)
    mask_gdb = gp.read_file(project["paths"]["subsubregions_path"])

    for geom in mask_gdb["geometry"]:
        local_mask = gp.GeoDataFrame({"geometry": [geom]}, crs=mask_gdb.crs)
        selection = gp.overlay(all_tiff_gdf, local_mask)
        if len(selection):
            print(list(selection["src_path"]))
            # fig, ax = plt.subplots()
            # all_tiff_gdf.plot(ax=ax, color="C0", alpha=0.2)
            # local_mask.plot(ax=ax, color="C1", alpha=0.2)
            # selection.plot(ax=ax, color="k")
            # plt.show()


if __name__ == "__main__":
    main()
