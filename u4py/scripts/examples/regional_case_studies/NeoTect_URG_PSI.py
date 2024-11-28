"""
Does a time series analysis for each of the regions of interest and saves the
plots in a figure.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import u4py.analysis.inversion as u4invert
import u4py.analysis.processing as u4proc
import u4py.io.files as u4files
import u4py.plotting.axes as u4ax
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~/Documents/umwelt4/Bericht_FFM_BBD_EGMS.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "piloten_path",
            "output_path",
            "processing_path",
        ],
        interactive=False,
    )
    overwrite = False

    # Setting up Paths
    bbd_fpath = os.path.join(
        project["paths"]["base_path"], "Data_2023", "hessen_l3_clipped.gpkg"
    )
    egms_fpath_vert1 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2015-2021",
        "EGMS_2015_2021_vertikal.gpkg",
    )
    egms_fpath_ew1 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2015-2021",
        "EGMS_2015_2021_ost_west.gpkg",
    )
    egms_fpath_vert2 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2018-2022",
        "EGMS_2018_2022_vertikal.gpkg",
    )
    egms_fpath_ew2 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2018-2022",
        "EGMS_2018_2022_ost_west.gpkg",
    )
    processing_path = os.path.join(
        project["paths"]["processing_path"], "FFM_EGMS_BBD"
    )
    results_log = []
    os.makedirs(processing_path, exist_ok=True)
    os.makedirs(project["paths"]["output_path"], exist_ok=True)
    # Loading Data
    regions = u4files.get_rois(project["paths"]["piloten_path"])
    for name, roi in tqdm(regions, desc="Reading Data and Creating Plots"):
        results_log.append(f"\n\n### Dataset: {name} ###\n\n")
        fig, axes = plt.subplots(
            figsize=(9, 5), nrows=2, sharex=True, sharey=True
        )
        results = u4proc.get_results(
            name,
            "BBD",
            roi,
            processing_path,
            [(bbd_fpath, "vertikal"), (bbd_fpath, "Ost_West")],
            overwrite=overwrite,
        )
        if results:
            make_plot(axes, results, color="C0")
            results_log.extend(fit_results("BBD", results))
        results = u4proc.get_results(
            name,
            "EGMS_1",
            roi,
            processing_path,
            [
                (egms_fpath_vert1, "EGMS_2015_2021_vertikal"),
                (egms_fpath_ew1, "EGMS_2015_2021_ost_west"),
            ],
            overwrite=overwrite,
        )
        if results:
            make_plot(axes, results, color="C1")
            results_log.extend(fit_results("EGMS_1", results))
        results = u4proc.get_results(
            name,
            "EGMS_2",
            roi,
            processing_path,
            [
                (egms_fpath_vert2, "EGMS_2018_2022_vertikal"),
                (egms_fpath_ew2, "EGMS_2018_2022_ost_west"),
            ],
            overwrite=overwrite,
        )
        if results:
            make_plot(axes, results, color="C2")
            results_log.extend(fit_results("EGMS_2", results))

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                project["paths"]["output_path"],
                f"{name}_inversion_timeseries",
            )
        )
        plt.close(fig)

    with open(
        os.path.join(processing_path, "Inversion_Log.txt"),
        "wt",
        encoding="utf8",
        newline="\n",
    ) as rsf:
        rsf.writelines(results_log)


def make_plot(axes: np.ndarray, results: dict, color: str):
    u4ax.plot_timeseries_fit(
        ax=axes[0], results=results, direction="U", color=color, color_fit="k"
    )
    u4ax.plot_timeseries_fit(
        ax=axes[1], results=results, direction="E", color=color, color_fit="k"
    )
    axes[0].set_ylabel("Vertical Position (mm)")
    axes[1].set_ylabel("Horizontal Position (mm)")
    axes[1].set_xlabel("Time")


def fit_results(data_name: str, results: dict):
    results_log = []
    results_log.append(f"\n\n - {data_name} - ")
    results_log.append("\nVertical")
    results_log.append(
        u4invert.print_inversion_results(
            results["U"]["inversion_results"],
            results["U"]["parameters_list"],
        )
    )
    results_log.append("\nEast-West")
    results_log.append(
        u4invert.print_inversion_results(
            results["E"]["inversion_results"],
            results["E"]["parameters_list"],
        )
    )
    return results_log


if __name__ == "__main__":
    main()
