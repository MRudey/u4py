""" Test the lookup procedure for inversion files"""

import os
from pathlib import Path

import geopandas as gp

import u4py.utils.files as u4files


def main():
    base_path = Path("~/Documents/ArcGIS").expanduser()
    pklfile_path = os.path.join(
        base_path, "INSAR_results", "inversion_results_2023_data.pkl"
    )
    pntfile_path = os.path.join(
        base_path, "selected_psi_points", "OilWell_points_BBD_Vert.shp"
    )
    points = gp.GeoDataFrame.from_file(pntfile_path)
    data, avg_dist = u4files.get_pickled_inversion_results(
        pklfile_path, points=points
    )
    print(points)
    print(data)
    print(f"Distance to points {avg_dist:.1f} m.")


if __name__ == "__main__":
    main()
