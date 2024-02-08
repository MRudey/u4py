import os

import geopandas as gp
import matplotlib.pyplot as plt

import u4py.io.files as u4files


def main():
    places_path = u4files.get_folder_paths(title="Select Places Folder")
    tektonik_path = os.path.join(places_path, "tektonik.shp")
    bld_path = os.path.join(places_path, "vg2500_bld.shp")

    tektonik = gp.read_file(tektonik_path).to_crs("EPSG:32632")
    bld = gp.read_file(bld_path).to_crs("EPSG:32632")
    hessen = bld[bld["GEN"] == "Hessen"]
    tektonik_hessen = gp.clip(tektonik, hessen)
    fig, ax = plt.subplots()
    tektonik_hessen.plot(ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
