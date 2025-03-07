from pathlib import Path

import geopandas as gp


def main():
    # in_file = Path("~/Documents/ArcGIS/SelectedSites_August24/Filtered_Classified_Shapes_all_merged_clean.gpkg").expanduser()
    # out_file = Path("~/Documents/ArcGIS/SelectedSites_August24/Filtered_Classified_Shapes_2410.gpkg").expanduser()
    # gdf: gp.GeoDataFrame
    # gdf = gp.read_file(in_file, where="area>20000 AND volumes_moved>10000")
    # gdf.to_file(out_file)

    in_file = Path(
        r"~\Documents\ArcGIS\SelectedSites_HLNUG_RD\RD_Rutschungen_gesamt.shp"
    ).expanduser()
    out_file = Path(
        r"~\Documents\ArcGIS\SelectedSites_HLNUG_RD\thresholded_contours_all_shapes.gpkg"
    ).expanduser()
    gdf: gp.GeoDataFrame
    gdf = gp.read_file(in_file)

    gdf = gdf[(gdf.OBJEKT != "keine Rutschung")]
    gdf = gp.GeoDataFrame(
        geometry=gdf.geometry,
        data={
            "groups": gdf.AMT_NR_,
        },
    )
    gdf.to_file(out_file)


if __name__ == "__main__":
    main()
