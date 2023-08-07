import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import u4py.utils.files as u4files


def main():
    file_path = u4files.get_file_paths(
        filetypes=((".h5", ".h5"),), title="Select file to plot."
    )
    out_dir = u4files.get_folder_paths(title="Select output folder.")
    os.makedirs(out_dir, exist_ok=True)

    for ud_file in file_path:
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
