import os

import fiona
import geopandas as gp
import matplotlib.pyplot as plt
import rasterio as rio
from tqdm import tqdm

import u4py.utils.files as u4files


def main():
    test_file = rio.open(
        r"D:\Projekte\Umwelt4\DGM-Differenzenplan_MitKorrektur\TB1\dgm1_32_468_5680_1_he_Kor_Diff_sys.tif",
        "r",
    )
    print(test_file.crs)

    # # shp_path = r"D:\Projekte\Umwelt4\hesse_shp\gis_osm_natural_free_1.shp"
    # # shp_path = r"D:\Projekte\Umwelt4\hesse_shp\gis_osm_buildings_a_free_1.shp"
    # # shp_path = r"D:\Projekte\Umwelt4\hesse_shp\lan-con.shp"

    # # data = gp.read_file(shp_path)
    # fclasses = ["quarry", "construction", "industrial"]
    # gdf = u4files.fiona_load(shp_path, fclasses)
    # # gdf.plot()
    # # plt.show()

    # print("Saving")
    # out_path = r"D:\Projekte\Umwelt4\hesse_shp\test.shp"
    # gdf.to_file(out_path)


if __name__ == "__main__":
    main()
