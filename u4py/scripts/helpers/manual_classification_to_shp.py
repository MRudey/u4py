"""
Uses the manual classification from a csv file to a shapefile for use with later classifications.
"""

import csv
import os
import re
from pathlib import Path

import geopandas as gp

import u4py.analysis.classify as u4class
import u4py.io.csvs as u4csvs


def main():
    file_path = Path(
        r"~\Documents\ArcGIS\SelectedSites_April24\manual_classification.csv"
    ).expanduser()
    man_class = get_manual_classes(file_path)

    gpkg_path = Path(
        r"~\Documents\ArcGIS\SelectedSites_April24\Classified_Shapes.gpkg"
    ).expanduser()
    gpkg_gdf = gp.read_file(gpkg_path)
    geometries = gpkg_gdf[gpkg_gdf.group.isin(man_class["group"])]

    new_gdf = gp.GeoDataFrame(
        data=man_class,
        geometry=geometries.geometry.centroid.to_list(),
        crs=gpkg_gdf.crs,
    )
    new_gdf.to_file(str(file_path).replace(".csv", ".shp"))

    # Testing classifier
    project = {
        "paths": {
            "sites_path": Path(
                r"~\Documents\ArcGIS\SelectedSites_April24"
            ).expanduser()
        }
    }
    res = dict()
    for geom in geometries.geometry.to_list():
        u4class.manual_classification(res, geom, "111", project)


def get_manual_classes(file_path: os.PathLike) -> dict:
    num_rows = u4csvs.buf_count_newlines_gen(file_path)
    with open(file_path, "rt", encoding="utf-8", newline="\n") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        header = next(reader)  # header currently not used
        data = {
            "group": [0 for ii in range(num_rows)],
            "known": [False for ii in range(num_rows)],
            "research": [False for ii in range(num_rows)],
            "class_1": ["" for ii in range(num_rows)],
            "unclear_1": [False for ii in range(num_rows)],
            "class_2": ["" for ii in range(num_rows)],
            "unclear_2": [False for ii in range(num_rows)],
            "class_3": ["" for ii in range(num_rows)],
            "unclear_3": [False for ii in range(num_rows)],
            "comment": ["" for ii in range(num_rows)],
        }
        for nrow, row in enumerate(reader):
            for ii, val in enumerate(row):
                if ii == 0:
                    data["group"][nrow] = int(re.sub("[^0-9]+", "", val))
                    if "\\underline" in val:
                        # To be examined more closely
                        data["research"][nrow] = True
                    if "\\textbf" in val:
                        # Previously known
                        data["known"][nrow] = True
                if ii == 1:
                    val = val.replace(" ", "")
                    classes = val.split(",")
                    for jj, cls in enumerate(classes):
                        if "$^{?}$" in cls:
                            data[f"unclear_{jj + 1}"][nrow] = True
                        cls = cls.replace("$^{?}$", "")
                        data[f"class_{jj + 1}"][nrow] = cls
                if ii == 2:
                    data["comment"][nrow] = (
                        val.replace("$^\\circ$", "°")
                        .replace("$^{?}$", "?")
                        .replace("$\\times$", "x")
                        .replace("\\,", "")
                    )
    return data


if __name__ == "__main__":
    main()
