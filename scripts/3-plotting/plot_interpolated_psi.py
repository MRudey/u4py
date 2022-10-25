import os

import matplotlib.pyplot as plt
import numpy as np
import u4py.utils.files as u4files
from scipy.interpolate import griddata


def main():
    ew_file = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Ost_West.h5"
    ud_file = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Vertikal.h5"
    out_dir = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_plots"

    os.makedirs(out_dir, exist_ok=True)
    x, y, z = get_data(ud_file)
    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 10000),
        np.linspace(np.min(y), np.max(y), 10000),
    )

    points = np.array([(xx, yy) for xx, yy in zip(x, y)])
    grid_1 = griddata(points, z, (grid_x, grid_y), method="cubic")
    fig, ax = plt.subplots()
    ax.imshow(grid_1)
    plt.show()


def get_data(ud_file, index=-2):
    data = u4files.load_hdf5(ud_file)
    z = data["timeseries"][:, index]
    ii = np.nonzero(np.isfinite(z))

    return (data["x"][ii], data["y"][ii], z[ii])


if __name__ == "__main__":
    main()
