import os
import shutil

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as spimage
from shapely.geometry import Point

import u4py.analysis.other as u4other
import u4py.analysis.spatial as u4spatial
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def main():
    gnss_folder = r"C:\Users\Michael Rudolf\Documents\ArcGIS\GNSS_Data"
    # get_converted_station_coordinates(gnss_folder)
    gnss_file_list = u4files.get_file_list(
        filetype=".dat", folder_path=gnss_folder
    )
    distance = 100
    filter_width = 7
    psivert_path = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Vertikal.h5"
    psiew_path = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Ost_West.h5"
    stations = get_converted_station_coordinates(gnss_folder)
    lookup_tree_vert = u4spatial.get_cKDTree(psivert_path)
    lookup_tree_ew = u4spatial.get_cKDTree(psiew_path)

    fig, axes = plt.subplots(nrows=len(stations))
    for ii, point in enumerate(stations.geometry):
        closest_vert = lookup_tree_vert.query_ball_point(point, distance)
        closest_vert.sort()
        closest_ew = lookup_tree_ew.query_ball_point(point, distance)
        closest_ew.sort()
        data_vert = u4files.load_hdf5(psivert_path, ind=closest_vert)
        data_ew = u4files.load_hdf5(psiew_path, ind=closest_ew)
        gnss_data = u4convert.gnss_dat_to_dict(gnss_file_list[ii])

        axes[ii].plot(
            gnss_data["gps_datetime"],
            spimage.median_filter(gnss_data["res_up"], filter_width),
            label="GPS-Data",
        )
        axes[ii].plot(
            data_vert["time"],
            np.nanmedian(data_vert["timeseries"], axis=0),
            label="PSI-Data (median)",
        )
    plt.show()
    # plot_gnss_and_psi(gnss_file_list)


def plot_gnss_and_psi(gnss_file_list):
    """Plots gnss and psi timeseries for all stations in gnss_file_list"""

    filter_width = 7
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey=True)
    i = 0
    for gnss_fpath in gnss_file_list:
        folder, fname = os.path.split(gnss_fpath)
        if os.path.exists(psivert_path):
            gnss_data = u4convert.gnss_dat_to_dict(gnss_fpath)
            # psi_vert_med = np.nanmedian(psi_vert["timeseries"], axis=0)
            # psi_ew = u4files.load_hdf5(psiew_path)
            # psi_ew_med = np.nanmedian(psi_ew["timeseries"], axis=0)

            axes[0][i].plot(
                gnss_data["gps_datetime"],
                spimage.median_filter(gnss_data["res_up"], filter_width),
                label="GPS-Data",
            )
            axes[0][i].plot(
                psi_vert["time"], psi_vert_med, label="PSI-Data (median)"
            )

            axes[1][i].plot(
                gnss_data["gps_datetime"],
                spimage.median_filter(gnss_data["res_east"], filter_width),
            )
            axes[1][i].plot(psi_ew["time"], psi_ew_med)

            axes[0][i].set_title(get_city_names(fname[:4]))
            i += 1
        else:
            pass
    plt.show()


def get_city_names(key):
    cities = {
        "BADH": "Bad Homburg",
        "FFMJ": "Frankfurt a.M.",
        "KLOP": "Kloppenheim / Frankfurt",
    }
    return cities[key]


def plot_cwt(x, y, f):
    freqs, coi, power = u4other.cwt(y, f)

    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(x, freqs)
    ax.contourf(xx, yy, power, levels=20)
    ax.fill_between(x, coi, np.min(freqs), color="w", alpha=0.25)
    plt.show()


def get_converted_station_coordinates(gnss_folder):
    stations = gp.GeoDataFrame(
        {
            "ID": ["BADH00DEU", "FFMJ00DEU", "KLOP00DEU"],
            "City": [
                "Bad Homburg",
                "Frankfurt a.M.",
                "Kloppenheim / Frankfurt",
            ],
            "geometry": [
                Point(8.6099, 50.228),
                Point(8.665, 50.0906),
                Point(8.7299, 50.2198),
            ],
        },
        crs="epsg:4326",
    )
    stations = stations.to_crs("EPSG:32632")
    # print("Station Data:")
    # print(stations)
    return stations


def copy_chunk(out_folder, psi_chunk, id):
    in_folder = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_chunks"

    folders = ["BBD_EW", "BBD_Vert"]
    for fol in folders:
        shutil.copyfile(
            os.path.join(in_folder, fol, psi_chunk),
            os.path.join(out_folder, f"{id}_{fol}.h5"),
        )


if __name__ == "__main__":
    main()
