import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import u4py.analysis.inversion as u4invert
import u4py.utils.files as u4files


def main():
    file_path = u4files.get_file_paths(filetypes=(("*.h5", "*.h5"),))
    time_ori, time, data, inversion_results = ts_inversion(file_path[0])
    plot_inversion_results(time_ori, data)


def plot_inversion_results(time, data):
    fig, axes = plt.subplots(
        ncols=2, figsize=(10, 5), sharex="row", sharey="row"
    )
    axes[0].set_title("East-West Component")
    axes[0].plot(time[0], data["dataE"], ".-", linewidth=0.5)
    axes[0].plot(time[0], data["ori_dhat_data"]["dhatE"])
    # axes[1][0].plot(time[0], data["ori_dhat_data"]["dhatE"], "C1")
    axes[1].set_title("Vertical Component")
    axes[1].plot(time[0], data["dataU"], ".-", linewidth=0.5)
    axes[1].plot(time[0], data["ori_dhat_data"]["dhatU"])
    # axes[1][1].plot(time[0], data["ori_dhat_data"]["dhatU"], "C1")
    fig.tight_layout()
    plt.show()


def ts_inversion(file_path="", mode="median", resample_data=2, maxn=4000):
    """Simultaneous inversion of multiple components of surface motion

    Args:
        file_path (str, optional): The filepath to the chunked and merged
            dataset. Defaults to "".
        mode (str, optional): Inversion mode. Defaults to "median".
        resample_data (int, optional): Data is resampled n amount of times to
            increase number of points (obeys fft upscaling). Defaults to 0.
            Only active for mode='median'
        maxn (int, optional): Maximum length of timeseries, longer are
            randomly sampled to the max length. Defaults to 2500.
    """
    if not file_path:
        time_series, t_EQ = u4invert.create_synthetic_data()
    else:
        dataset = u4files.load_hdf5(file_path, timefmt="floatyear")
        data_keys = [k for k in dataset.keys()]

        # Check if steps are defined in the dataset
        if "t_EQ" in data_keys:
            t_EQ = dataset["t_EQ"]
        else:
            t_EQ = []

        # Averages data in chunks
        if mode == "single":
            time_series = dataset[data_keys[0]]
        elif mode == "median":
            time_series = u4invert.medianize_station(
                dataset, data_keys, include_sigma=False
            )
        elif mode == "stack":
            time_series = u4invert.stack_data(dataset, data_keys)

        if resample_data and mode == "median":
            for k in ["dataE", "dataN", "dataU"]:
                ind = np.nonzero(np.isfinite(time_series[k]))
                time_series[k] = spsig.resample(
                    time_series[k][ind],
                    resample_data * len(time_series[k]),
                )[:-resample_data]
                time_series["t"] = np.linspace(
                    np.min(dataset[data_keys[0]]["t"]),
                    np.max(dataset[data_keys[0]]["t"]),
                    len(dataset[data_keys[0]]["t"]) * resample_data,
                )[:-resample_data]

            time_series["xmid"] = dataset[data_keys[0]]["xmid"]
            time_series["ymid"] = dataset[data_keys[0]]["ymid"]

        if len(time_series["dataE"]) > maxn:
            time_series = u4invert.downsample_timeseries(time_series, maxn)

        if "sigmE" not in time_series.keys():
            time_series["sigmE"] = np.ones_like(time_series["dataE"])
            time_series["sigmN"] = np.ones_like(time_series["dataE"])
            time_series["sigmU"] = np.ones_like(time_series["dataE"])

    matrix_ori, data, time_vector_ori = u4invert.invert_time_series(
        time_series, t_EQ=t_EQ
    )
    ind = u4invert.remove_outliers(data["ori_dhat_data"])
    matrix, data, time_vector = u4invert.invert_time_series(
        time_series, ind=ind
    )

    inversion_results = {"matrix_ori": matrix_ori, "matrix": matrix}
    return time_vector_ori, time_vector, data, inversion_results


if __name__ == "__main__":
    main()
