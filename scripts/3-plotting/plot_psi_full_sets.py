import os
from datetime import datetime
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import u4py.utils.files as u4files


def main():
    file_path = u4files.get_file_paths(
        filetypes=((".h5", ".h5"),), title="Select file to plot."
    )
    out_dir = u4files.get_folder_paths(title="Select output folder.")
    os.makedirs(out_dir, exist_ok=True)

    for ud_file in file_path:
        os.makedirs(out_dir, exist_ok=True)
        data = u4files.load_hdf5(ud_file)
        skips = 25
        vm = np.max(
            np.abs(
                (
                    np.nanpercentile(data["timeseries"], 5),
                    np.nanpercentile(data["timeseries"], 95),
                )
            )
        )
        for ii in tqdm(range(len(data["time"]))):
            # ii = 1
            plot_data(
                data["x"][::skips],
                data["y"][::skips],
                data["timeseries"][::skips, ii],
                data["time"][ii],
                out_dir,
                vm=vm,
                ii=ii,
            )


def plot_data(
    x: np.array,
    y: np.array,
    z: np.array,
    t: datetime,
    outdir=PathLike,
    vm=float,
    ii=int,
):
    ii += 1
    out_path = os.path.join(
        outdir, f"{ii:03}"
    )  # + t.strftime("scatter_%Y%m%d"))
    fig, ax = plt.subplots(dpi=150)
    scat = ax.scatter(
        x, y, c=z, marker=".", s=5, vmin=-vm, vmax=vm, cmap="RdBu_r"
    )
    plt.colorbar(scat, ax=ax)
    ax.annotate(
        t.strftime("%d.%m.%Y"),
        (0.95, 0.05),
        xycoords="axes fraction",
        horizontalalignment="right",
    )
    ax.axis("equal")
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
