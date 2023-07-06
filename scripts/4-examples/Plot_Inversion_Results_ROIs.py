"""
Plots the inversion results for each of the regions of interest including
tectonic information.
"""
import os
import pickle as pkl

from tqdm import tqdm

import u4py.plotting.plots as u4plots
import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        required=[
            "base_path",
            "base_map_path",
            "tektonik_path",
            "piloten_path",
            "output_path",
            "results_path",
        ]
    )

    os.makedirs(project["paths"]["output_path"], exist_ok=True)

    data = load_data(project["paths"]["results_path"])
    converted_data = u4plotprep.convert_results_for_grid(data[0])
    lin_2d, sin_2d, extend = u4plotprep.make_gridded_data(*converted_data)
    regions = u4files.get_rois(project["paths"]["piloten_path"])
    for name, roi in tqdm(regions, desc="Generating plots from regions"):
        u4plots.plot_gridded(
            lin_2d,
            sin_2d,
            extend,
            roi=roi,
            tektonik_path=project["paths"]["tektonik_path"],
            base_map_path=project["paths"]["base_map_path"],
            dpi=150,
            save_path=os.path.join(
                project["paths"]["output_path"], f"roi_{name}_inversions"
            ),
        )


def load_data(file_path):
    """Loads data from selected pickle file"""
    with open(file_path, "rb") as pklfile:
        data = pkl.load(pklfile)
    return data


if __name__ == "__main__":
    main()
