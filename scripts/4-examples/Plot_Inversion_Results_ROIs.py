"""
Plots the inversion results for each of the regions of interest including
tectonic information.
"""
import os
import pickle as pkl

from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.plotting.plots as u4plots
import u4py.plotting.preparation as u4plotprep
import u4py.utils.files as u4files


def main():
    file_path = u4files.get_file_paths(
        filetypes=((".pkl", ".pkl"),), title="Select inversion results file."
    )

    base_path = u4files.get_folder_paths(title="Select basepath.")

    base_map_path = os.path.join(base_path, "Places", "hessen_map.tif")
    tektonik_path = os.path.join(base_path, "Places", "tektonik_cropped.shp")
    # piloten_path = os.path.join(base_path, "Places", "Pilotregionen.shp")
    piloten_path = os.path.join(base_path, "Places", "FFM_regions.shp")
    output_path = os.path.join(base_path, "INSAR_plots")
    os.makedirs(output_path, exist_ok=True)

    data = load_data(file_path)
    converted_data = u4plotprep.convert_results_for_grid(data[0])
    lin_2d, sin_2d, extend = u4plotprep.make_gridded_data(*converted_data)
    regions = u4spatial.get_rois(piloten_path)
    for name, roi in tqdm(regions, desc="Generating plots from regions"):
        u4plots.plot_gridded(
            lin_2d,
            sin_2d,
            extend,
            roi=roi,
            tektonik_path=tektonik_path,
            base_map_path=base_map_path,
            dpi=150,
            save_path=os.path.join(output_path, f"roi_{name}_inversions"),
        )


def load_data(file_path):
    """Loads data from selected pickle file"""
    with open(file_path, "rb") as pklfile:
        data = pkl.load(pklfile)
    return data


if __name__ == "__main__":
    main()
