"""
Merges Point Shapefiles and removes duplicate Points
"""
import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt


def main():
    folder_path = Path(
        r"~\Documents\ArcGIS\Places\ADA_Shapes\Erdfall"
    ).expanduser()
    folder_path_all = os.path.join(folder_path, "AllPoints")
    file_list = [
        os.path.join(folder_path_all, fp)
        for fp in os.listdir(folder_path_all)
        if fp.endswith(".shp")
    ]
    geometries = []
    for shp_file in file_list:
        gdf = gp.read_file(shp_file)
        if gdf.crs != "EPSG:32632":
            gdf = gdf.to_crs("EPSG:32632")
        geometries.extend(gdf.geometry.to_wkt())
    print("Total Points:", len(geometries))
    geom_series = gp.GeoSeries.from_wkt(geometries).buffer(10)
    merged_polygons = gp.GeoDataFrame(
        geometry=[
            gp.GeoDataFrame(geometry=geom_series, crs="EPSG:32632").unary_union
        ],
        crs="EPSG:32632",
    ).explode(index_parts=True)
    merged_points = gp.GeoDataFrame(
        geometry=merged_polygons.centroid, crs=gdf.crs
    )
    print("Merged Points:", len(merged_points))
    merged_points.to_file(os.path.join(folder_path, "Erdfaelle_merged.shp"))
    # fig, ax = plt.subplots()
    # merged_points.plot(ax=ax, marker=".")
    # plt.show()


if __name__ == "__main__":
    main()
