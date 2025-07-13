"""
Merges the manual classification done by M. Koubik into a single file.
"""

import os
from pathlib import Path

import geopandas as gp


def main():
    base_path = Path(
        "~/Documents/umwelt4/Places/Bewegungsatlas 2024 Koubik"
    ).expanduser()

    rm_path = os.path.join(base_path, "Klassifikation_RheinMain.shp")
    wk_path = os.path.join(base_path, "Klassifikation_WerraKali.shp")
    rm_gdf = convert_to_manual_class(rm_path)
    wk_gdf = convert_to_manual_class(wk_path)

    merged = gp.pd.concat([rm_gdf, wk_gdf])

    merged.to_file(os.path.join(base_path, "koubik_merged.shp"))


def convert_to_manual_class(in_file: os.PathLike | str) -> gp.GeoDataFrame:
    """Converts the input file into a geodataframe with the corrrect data keys.

    :param in_file: The input shapefile
    :type in_file: os.PathLike | str
    :return: The geodataframe with correct data keys.
    :rtype: gp.GeoDataFrame
    """
    translate = {
        "Abbau": "Steinbruch",
        "Baumaßnahme": "Baustelle",
        "Deponie": "Deponie",
        "Flughafen": "Industrie",
        "Freibad": "Freizeitanlage",
        "Gewässer": "Gewässer",
        "Golfplatz": "Freizeitanlage",
        "Hangbewegung": "Rutschung",
        "Landwirtschaft": "Landwirtschaft",
        "Parkplatz": "Verkehr",
        "Pflanzenwachstum": "Landwirtschaft",
        "Sportplatz": "Freizeitanlage",
        "unbekannt": "Unbekannt",
    }

    gdf = gp.read_file(in_file)
    num_rows = len(gdf)
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
    for ii, row in enumerate(gdf.iterrows()):
        data["group"][ii] = ii
        clss = row[1]["Klasse"].replace("(", "").replace(")", "")
        data["class_1"][ii] = translate[clss]
        if "(" in row[1]["Klasse"]:
            data["unclear_1"][ii] = True
        if "unbekannt" in clss:
            data["research"][ii] = True
        data["comment"][ii] = row[1]["Bezeichnun"]

    out_gdf = gp.GeoDataFrame(data=data, geometry=gdf.geometry, crs=gdf.crs)
    return out_gdf


if __name__ == "__main__":
    main()
