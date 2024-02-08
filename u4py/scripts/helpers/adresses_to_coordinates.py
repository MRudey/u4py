"""
Script to load coordinates of all high-rise buildings in FFM
(according to wikipedia data in CSV file)
"""

import csv
from functools import partial
from pathlib import Path

import geopandas as gp
import shapely
from geopy.geocoders import Nominatim
from tqdm import tqdm

geolocator = Nominatim(user_agent="http")


def main():
    file_path = Path(
        "~/Documents/ArcGIS/Places/Highrise_buildings.csv"
    ).expanduser()
    geocode = partial(geolocator.geocode, language="de")

    name = []
    height = []
    num_st = []
    opening = []
    address = []
    lat = []
    lon = []
    geometry = []
    wrong_list = []

    with open(file_path, "rt", encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in tqdm(csv_reader):
            query = line[5].replace(".", "") + " Frankfurt"
            geom = geocode(query)
            if geom:
                lat.append(geom.latitude)
                lon.append(geom.longitude)
                name.append(line[1])
                try:
                    height.append(float(line[2]))
                except ValueError:
                    height.append(0.0)
                try:
                    num_st.append(int(line[3]))
                except ValueError:
                    num_st.append(0)
                try:
                    opening.append(int(line[4]))
                except ValueError:
                    opening.append(0)
                address.append(line[5])
                geometry.append(shapely.Point(geom.longitude, geom.latitude))
            else:
                wrong_list.append(f"{query}")

    gdb = gp.GeoDataFrame(
        data={
            "name": name,
            "height": height,
            "num_st": num_st,
            "opening": opening,
            "address": address,
        },
        geometry=geometry,
        crs="EPSG:4326",
    )
    gdb.to_file(file_path.replace(".csv", ".shp"))

    print(wrong_list)


if __name__ == "__main__":
    main()
