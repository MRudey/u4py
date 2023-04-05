"""
Plots the results from inversion processing that has been saved as a pickle
file.
"""

import os
import pickle as pkl

import u4py.plotting.plots as u4plots
import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files


def main():
    file_path = u4files.get_file_paths(
        filetypes=((".pkl", ".pkl"),), title="Select inversion results file."
    )

    base_path = u4files.get_folder_paths(title="Select basepath.")

    base_map_path = os.path.join(base_path, "Places", "hessen_map.tif")
    tektonik_path = os.path.join(base_path, "Places" "tektonik_cropped.shp")

    for fp in file_path:
        data = load_data(fp)
        converted_data = u4plotprep.convert_results_for_grid(data[0])
        lin_2d, sin_2d, extend = u4plotprep.make_gridded_data(*converted_data)
        u4plots.plot_gridded(
            lin_2d,
            sin_2d,
            extend,
            tektonik_path=tektonik_path,
            base_map_path=base_map_path,
            dpi=100,
        )


def load_data(file_path):
    """Loads data from selected pickle file"""
    with open(file_path, "rb") as pklfile:
        data = pkl.load(pklfile)
    return data


if __name__ == "__main__":
    main()
