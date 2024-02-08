"""Creates quick overview of detected ground motions"""

import os
import pickle as pkl
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np

import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    overwrite = True
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Playground\StatisticsDetectedMotions_gpkg.u4project"
        ).expanduser(),
        required=["places_path", "output_path", "processing_path"],
        interactive=False,
    )
    out_folder = os.path.join(project["paths"]["output_path"], "Playground")
    os.makedirs(out_folder, exist_ok=True)

    stat_file_path = os.path.join(
        project["paths"]["processing_path"], "thresholded_contours_stats.pkl"
    )

    if not os.path.exists(stat_file_path) or overwrite:
        shape_path = os.path.join(
            project["paths"]["places_path"],
            "thresholded_contours_all_shapes.gpkg",
        )
        shapes = gp.read_file(shape_path)
        shape_data = dict()
        for k in shapes.keys():
            if k != "geometry":
                shape_data[k] = shapes[k].to_numpy()
        with open(stat_file_path, "wb") as stat_file:
            pkl.dump(shape_data, stat_file)
    else:
        with open(stat_file_path, "rb") as stat_file:
            shape_data = pkl.load(stat_file)

    # Some Post-Processing
    slc = np.abs(shape_data["avg_displ"]) > 0.05

    print(
        len(shape_data["areas"][slc]),
        "of",
        len(shape_data["areas"]),
        "have more than 5 cm displacement.",
    )
    # Plotting
    fig, ax = plt.subplots(dpi=150)
    ax.plot(shape_data["areas"][slc], shape_data["sums"][slc], ".")
    ax.set_xscale("log")
    ax.set_xlabel("Area (m$^2$)")
    ax.set_ylabel("Volume (m$^3$)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "DetectedMotions_AreaVolumes"))
    fig.savefig(os.path.join(out_folder, "DetectedMotions_AreaVolumes.pdf"))

    fig, ax = plt.subplots(dpi=150)
    ax.plot(shape_data["areas"][slc], shape_data["avg_displ"][slc], ".")
    ax.set_xscale("log")
    ax.set_xlabel("Area (m$^2$)")
    ax.set_ylabel("avg. Displacement (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "DetectedMotions_AreaDisplacement"))
    fig.savefig(
        os.path.join(out_folder, "DetectedMotions_AreaDisplacement.pdf")
    )

    fig, ax = plt.subplots(dpi=150)
    highs = np.percentile(shape_data["avg_displ"], 95)
    ax.hist(shape_data["avg_displ"], bins=np.linspace(-highs, highs, 100))
    ax.set_xlabel("avg. Displacement (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "DetectedMotions_Displacement_hist"))
    fig.savefig(
        os.path.join(out_folder, "DetectedMotions_Displacement_hist.pdf")
    )


if __name__ == "__main__":
    main()
