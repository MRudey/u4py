import os
from datetime import datetime
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import u4py.utils.files as u4files
from tqdm import tqdm


def main():
    ew_file = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Ost_West.h5"
    ud_file = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Vertikal.h5"
    out_dir = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_plots"

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
