"""
Finds regions according to the classification criteria in merged tiffs.
"""

import logging
import os
from pathlib import Path

# import matplotlib.pyplot as plt
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial

# import u4py.plotting.formatting as u4pltfmt
import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    project = u4projects.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/Find_Regions_in_Merged_Tif_Server.u4project"
        ).expanduser(),
        required=["diff_plan_path"],
    )

    threshold = 250  # => Tree with approx. 17.8 m diameter
    levels = u4spatial.plus_minus_levels([0.5, 1, 2, 5, 10])

    file_list = u4files.get_file_list(
        folder_path=project["paths"]["diff_plan_path"], filetype=".tif"
    )
    for fp in tqdm(file_list):
        fname = os.path.split(fp)
        logging.info(f"Reading {fname}")
        try:
            gdf = u4files.get_thresholded_contours(
                fp, levels, threshold, overwrite=True
            )
        except:
            pass
    #     fig, ax = plt.subplots()
    #     gdf.plot(ax=ax)
    #     u4pltfmt.map_style(ax)
    # plt.show()


if __name__ == "__main__":
    main()
