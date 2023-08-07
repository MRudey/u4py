"""
Converts the CSV File by the HLNUG to a pickled dictionary and a shapefile
with all points.
"""

import os
import pickle as pkl
from pathlib import Path

import u4py.addons.groundwater as u4gw


def main():
    file_path = Path(
        "~/Documents/ArcGIS/ExternalData/GWStände_2015/GWStände_2015.CSV"
    ).expanduser()
    output_file, _ = os.path.splitext(file_path)
    stations = u4gw.convert_GW_csv(file_path)
    with open(output_file + ".pkl", "wb") as pkl_file:
        pkl.dump(stations, pkl_file)

    stations_gdf = u4gw.get_stations(stations)
    stations_gdf.to_file(
        Path(
            "~/Documents/ArcGIS/ExternalData/GWStände_2015/GW_Stations.shp"
        ).expanduser()
    )


if __name__ == "__main__":
    main()
