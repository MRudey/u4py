"""Script to detect ground motion in the PSI data set using the
GroundMotionAnalyzer, a variance based filtering algorithm.
"""

import os
from pathlib import Path

import u4py.plotting.plots as u4plots

# import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

# u4config.start_logger()


def main():
    # Parameters
    cellsize = [100, 250, 500, 1000]  # (m) cell size of binning
    min_mean = [0.5, 1, 1.5, 2, 2.5]  # (mm/a) minimum velocity
    max_var = [0.5, 1, 1.5, 2, 3, 4, 5]  # () maximum variance

    # cellsize = 500  # (m) cell size of binning
    # min_mean = 2  # (mm/a) minimum velocity
    # max_var = [0.5, 1, 2, 3, 4, 5]  # () maximum variance

    # Paths
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\GroundMotionAnalyzer.u4project"
        ).expanduser(),
        required=["base_path", "psi_path", "output_path", "processing_path"],
        interactive=False,
    )
    make_plot(project, cellsize, min_mean, max_var)


def make_plot(project, cellsize, min_mean, max_var):
    psi_fpath = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
    )
    out_folder = os.path.join(project["paths"]["output_path"], "GMA")
    os.makedirs(out_folder, exist_ok=True)
    output_filepath = os.path.join(
        out_folder,
        f"GMA_Python_cell={cellsize}_minmean={min_mean}_maxvar={max_var}.png",
    )

    # Analysis and Plot
    u4plots.plot_GroundMotionAnalyzer(
        psi_fpath,
        project["paths"]["processing_path"],
        cellsize=cellsize,
        min_mean=min_mean,
        max_var=max_var,
        # output_filepath=output_filepath,
        overwrite=True,
    )


if __name__ == "__main__":
    main()
