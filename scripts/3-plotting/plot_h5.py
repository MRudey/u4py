import numpy as np

# import scipy.interpolate as spint
import scipy.signal as spsig
import u4py.utils.files as u4files
from matplotlib import pyplot as plt


def main():
    """
    main function
    """
    file_list = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    for file_path in file_list:
        data = u4files.load_hdf5(file_path, timefmt="floatyear")
        time = data["time"]
        y = np.nanmedian(data["timeseries"], axis=0)
        ind = np.nonzero(np.isfinite(y))
        num_samples = 5 * len(y)
        # spline = spint.CubicSpline(time[ind], y[ind])
        y_resampled = spsig.resample(y[ind], num_samples)

        time_q = np.linspace(np.min(time), np.max(time), num_samples)
        fig, ax = plt.subplots()
        ax.plot(time, y, "o-")
        # ax.plot(time_q, spline(time_q))
        ax.plot(time_q, y_resampled, ".-", linewidth=0.5)
        plt.show()


if __name__ == "__main__":
    main()
