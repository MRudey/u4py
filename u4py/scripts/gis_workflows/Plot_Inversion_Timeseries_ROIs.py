"""
Gets PSI points within all rectangles in the shape file, does an inversion and plots the results.

Saves the results for each rectangle in a separate pkl file.
"""

import os

import matplotlib.pyplot as plt
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.plotting.axes as u4ax
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        required=[
            "psivert_path",
            "base_path",
            "piloten_path",
            "output_path",
            "processing_path",
        ],
        interactive=False,
    )

    regions = u4files.get_rois(project["paths"]["piloten_path"])
    for name, roi in tqdm(regions, desc="Inverting and plotting"):
        points, _ = u4files.get_region_points(
            roi, name, project["paths"]["psivert_path"]
        )
        data = u4files.load_data_from_points(
            project["paths"]["psivert_path"], points
        )
        inv_save_path = os.path.join(
            project["paths"]["processing_path"], f"{name}.pkl"
        )
        results = u4proc.invert_psi_dict(data, save_path=inv_save_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        u4ax.plot_timeseries_fit(ax=ax, results=results)
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Vertical Position (mm)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                project["paths"]["output_path"], f"{name}_inversion_timeseries"
            )
        )


if __name__ == "__main__":
    main()
