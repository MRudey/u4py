import os

import numpy as np

# import scipy.interpolate as spint
import scipy.signal as spsig
from matplotlib import pyplot as plt

import u4py.utils.files as u4files


def main():
    """
    main function
    """
    file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    for file_path in file_list:
        # Setting paths
        folder_path, fname = os.path.split(file_path)
        base_path, _ = os.path.split(folder_path)
        chunk, _ = os.path.splitext(fname)
        out_path = os.path.join(base_path, chunk + ".png")

        # Loading data and do some calculations
        data = u4files.load_hdf5(file_path, timefmt="floatyear")
        time = data["time"]
        y = np.nanmedian(data["timeseries"], axis=0)
        ind = np.nonzero(np.isfinite(y))
        num_samples = 5 * len(y)
        # spline = spint.CubicSpline(time[ind], y[ind])
        y_resampled = spsig.resample(y[ind], num_samples)
        time_q = np.linspace(np.min(time), np.max(time), num_samples)

        # Make Plot
        fig, ax = plt.subplots()
        ax.plot(time, y, "o-")
        # ax.plot(time_q, spline(time_q))
        ax.plot(time_q, y_resampled, ".-", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_path)


if __name__ == "__main__":
    main()
