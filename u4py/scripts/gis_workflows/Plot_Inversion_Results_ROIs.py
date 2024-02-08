"""
Plots the inversion results for each of the regions of interest including
tectonic information.
"""
import os
import pickle as pkl

from tqdm import tqdm

import u4py.io.files as u4files
import u4py.plotting.plots as u4plots
import u4py.plotting.preparation as u4plotprep
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
    _, fname = os.path.split(project["paths"]["results_path"])
    if "2021_old" in fname:
        yr = "2021_old"
    elif "2021" in fname:
        yr = "2021"
    elif "2023" in fname:
        yr = "2023"
    save_path = os.path.join(
        project["paths"]["output_path"], "regions_of_interest"
    )
    os.makedirs(save_path, exist_ok=True)
    converted_data, chunk_size = u4plotprep.convert_results_for_grid(
        data[0], chunk_size=100
    )
    grids, extend = u4plotprep.make_gridded_data(converted_data, chunk_size)
    regions = u4files.get_rois(project["paths"]["piloten_path"])
    for name, roi in tqdm(regions, desc="Generating plots from regions"):
        u4plots.plot_gridded(
            grids[0],
            grids[1],
            extend,
            roi=roi,
            tektonik_path=project["paths"]["tektonik_path"],
            base_map_path=project["paths"]["base_map_path"],
            dpi=150,
            save_path=os.path.join(
                save_path, f"roi_{name}_{yr}_{chunk_size}m_inv"
            ),
        )


def load_data(file_path):
    """Loads data from selected pickle file"""
    with open(file_path, "rb") as pklfile:
        data = pkl.load(pklfile)
    return data


if __name__ == "__main__":
    main()
