"""Creates quick overview of detected ground motions"""

import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np

import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Playground\StatisticsDetectedMotions.u4project"
        ).expanduser(),
        required=["processing_path", "piloten_path", "output_path"],
        interactive=False,
    )
    out_folder = os.path.join(project["paths"]["output_path"], "Playground")
    os.makedirs(out_folder, exist_ok=True)

    file_list = u4files.get_file_list(
        filetype=".shp", folder_path=project["paths"]["processing_path"]
    )
    piloten_shp = gp.read_file(project["paths"]["piloten_path"])
    piloten_areas = dict()
    for name, area in zip(
        piloten_shp.Name.to_numpy(), piloten_shp.area.to_numpy()
    ):
        piloten_areas[name.replace(" ", "")] = area
    fig, axes = plt.subplots(nrows=3, sharex=True)
    maxx = 0
    for ii, shp_file in enumerate(file_list):
        fname = (
            os.path.splitext(os.path.split(shp_file)[-1])[0]
            .replace("thresh_contours_", "")
            .replace("_merged", "")
        )
        shapes = gp.read_file(shp_file)
        levels = shapes.level.to_numpy()
        area = shapes.area.to_numpy()
        x = np.unique(levels)
        if np.max(np.abs(x)) > maxx:
            maxx = np.max(np.abs(x))
        y = []
        areas = []
        perc = []
        for ulvl in np.unique(levels):
            slc = np.argwhere(levels == ulvl)
            y.append(len(slc))
            areas.append(np.sum(area[slc]) / (1000**2))
            perc.append((np.sum(area[slc]) / piloten_areas[fname]) * 100)
        axes[0].plot(
            x,
            y,
            "s",
            markeredgecolor=f"C{ii}",
            markerfacecolor="None",
            label=fname,
        )
        axes[1].plot(
            x,
            areas,
            "s",
            markeredgecolor=f"C{ii}",
            markerfacecolor="None",
        )
        axes[2].plot(
            x,
            perc,
            "s",
            markeredgecolor=f"C{ii}",
            markerfacecolor="None",
        )
    axes[0].set_xlim(-maxx - 1, maxx + 1)
    axes[0].set_ylabel("Number of Features ()")
    axes[1].set_ylabel("Total Area (km²)")
    axes[2].set_ylabel("Relative Area (%)")
    axes[-1].set_xlabel("Ground Motion Class (m)")
    # axes[0].set_yscale("log")
    # axes[1].set_yscale("log")
    axes[0].legend(loc="best", fontsize="small", markerscale=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "StatisticsDetectedMotion.png"))
    fig.savefig(os.path.join(out_folder, "StatisticsDetectedMotion.pdf"))


if __name__ == "__main__":
    main()
