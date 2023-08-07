"""
Plots raw data from the groundwater database.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import u4py.addons.groundwater as u4gw
import u4py.plotting.axes as u4ax


def main():
    gw_file_path = Path(
        "~/Documents/ArcGIS/ExternalData/GWStände_2015/GWStände_2015.pkl"
    ).expanduser()
    stations = u4gw.get_groundwater_data(gw_file_path)

    # for k in stations.keys():
    #     print(k)
    fig, ax = plt.subplots()
    ax.plot(stations["CRUMSTADT"]["time"], stations["CRUMSTADT"]["height"])
    plt.show()


if __name__ == "__main__":
    main()
