"""Script to detect ground motion in the PSI data set using the
GroundMotionAnalyzer, a variance based filtering algorithm.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

import u4py.plotting.plots as u4plots
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    # Parameters
    cellsize = 500  # (m) cell size of binning
    min_mean = 2  # (mm/a) minimum velocity
    max_var = 1  # () maximum variance

    # Paths
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\GroundMotionAnalyzer.u4project"
        ).expanduser(),
        required=["base_path", "psi_path", "output_path", "processing_path"],
        interactive=False,
    )
    psi_fpath = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
    )
    output_filepath = os.path.join(
        project["paths"]["output_path"],
        f"GMA_Python_cell={cellsize}_minmean={min_mean}_maxvar={max_var}.png",
    )

    # Analysis and Plot
    u4plots.plot_GroundMotionAnalyzer(
        psi_fpath,
        project["paths"]["processing_path"],
        cellsize=cellsize,
        min_mean=min_mean,
        max_var=max_var,
        output_filepath=output_filepath,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
