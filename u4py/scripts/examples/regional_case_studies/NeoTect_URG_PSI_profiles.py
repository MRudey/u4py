import os
import pickle as pkl
from pathlib import Path

import matplotlib.axes as maxs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.io.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Examples\NeotectonicsURG_FaultMovement.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "piloten_path",
            "output_path",
            "processing_path",
        ],
        interactive=False,
    )
    overwrite = False

    # Setting up Paths
    bbd_fpath = os.path.join(
        project["paths"]["base_path"], "Data_2023", "hessen_l3_clipped.gpkg"
    )
    egms_fpath_vert1 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2015-2021",
        "EGMS_2015_2021_vertikal.gpkg",
    )
    egms_fpath_ew1 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2015-2021",
        "EGMS_2015_2021_ost_west.gpkg",
    )
    egms_fpath_vert2 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2018-2022",
        "EGMS_2018_2022_vertikal.gpkg",
    )
    egms_fpath_ew2 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2018-2022",
        "EGMS_2018_2022_ost_west.gpkg",
    )
    processing_path = os.path.join(
        project["paths"]["processing_path"], "NeoTect_URG_FaultMovements"
    )
    results_log = []
    os.makedirs(processing_path, exist_ok=True)

    fignum = {
        "1A": 0,
        "1B": 1,
        "1C": 2,
        "1D": 3,
        "1E": 4,
        "2A": 5,
        "2B": 6,
    }
    annotations = {
        "1A": [("Worms", 1500), ("Bahn", 4500), ("Wattenheim", 6500)],
        "1B": [("Hofheim", 500), ("Nordheim", 2500)],
        "1C": [("Wattenheim", 500), ("Rheinaue", 2000), ("Ibersheim", 3500)],
        "1D": [
            ("Kleinrohrheim", 500),
            ("Gernsheim", 2500),
            ("Biebesheim", 5500),
            ("Stockstadt", 8500),
            ("Erfelden", 11500),
            ("Hof Hayna", 13500),
        ],
        "1E": [
            ("Allmendfeld", 1000),
            ("Biebesheim", 3800),
            ("Crumstadt", 6300),
            ("Goddelau", 8300),
            ("Riedstadt", 11000),
        ],
        "2A": [("Leeheim", 500), ("Riedstadt", 2500), ("Goddelau", 3000)],
        "2B": [("Stockstadt", 2000)],
    }
    figures = []
    axes = []
    for ii in range(7):
        fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            sharex="all",
            sharey="all",
            figsize=(12, 6),
            dpi=150,
        )
        figures.append(fig)
        axes.append(axs)
    surveys = ["BBD", "EGMS_1", "EGMS_2"]
    direct_paths = {
        "BBD": [(bbd_fpath, "vertikal"), (bbd_fpath, "Ost_West")],
        "EGMS_1": [
            (egms_fpath_vert1, "EGMS_2015_2021_vertikal"),
            (egms_fpath_ew1, "EGMS_2015_2021_ost_west"),
        ],
        "EGMS_2": [
            (egms_fpath_vert2, "EGMS_2018_2022_vertikal"),
            (egms_fpath_ew2, "EGMS_2018_2022_ost_west"),
        ],
    }
    regions = u4files.get_rois(project["paths"]["piloten_path"])
    for name, roi in tqdm(regions, desc="Regions", leave=False):
        axs = axes[fignum[name[:2]]]
        if "1" in name:  # Select along which coordinate to plot
            drc = "y"
            bidx = 1
        else:
            drc = "x"
            bidx = 0
        for ii, surv in enumerate(tqdm(surveys, desc="Surveys", leave=False)):
            profile = u4proc.extract_profile(
                name,
                surv,
                roi,
                processing_path,
                direct_paths[surv],
                overwrite=overwrite,
            )
            if "West" in name or "Nord" in name:
                jj = 0
            else:
                jj = 1

            add_profile(
                axs[0][jj],
                # axes[0 + jj][axnum[name[:2]]],
                profile[f"{drc}_v"],
                profile["vert"],
                ii,
                xoffset=roi.bounds[bidx],
            )
            add_profile(
                axs[1][jj],
                # axes[1 + jj][axnum[name[:2]]],
                profile[f"{drc}_ew"],
                profile["ew"],
                ii,
                xoffset=roi.bounds[bidx],
            )
        axs[0][0].set_ylim(-3, 3)
        axs[0][0].set_xlim(0, roi.bounds[bidx + 2] - roi.bounds[bidx])
        axs[0][jj].set_title(name[3:])
        if bidx:
            for kk in range(2):
                axs[0][kk].annotate(
                    "Süden",
                    (0, 1.05),
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
                axs[0][kk].annotate(
                    "Norden",
                    (1, 1.05),
                    xycoords="axes fraction",
                    horizontalalignment="right",
                )
        else:
            for kk in range(2):
                axs[0][kk].annotate(
                    "Westen",
                    (0, 1.05),
                    xycoords="axes fraction",
                    horizontalalignment="left",
                )
                axs[0][kk].annotate(
                    "Osten",
                    (1, 1.05),
                    xycoords="axes fraction",
                    horizontalalignment="right",
                )
        axs[0][0].set_ylabel("Vertical (mm/yr)")
        axs[1][0].set_ylabel("East-West (mm/yr)")
        axs[1][0].set_xlabel("Along Strike (m)")
        axs[1][1].set_xlabel("Along Strike (m)")
        annot = annotations[name[:2]]
        for ann in annot:
            for kk in range(2):
                axs[0][kk].annotate(
                    ann[0],
                    (ann[1], 2.5),
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize="small",
                    fontstyle="italic",
                    rotation=25,
                )
        for ax in axs.flat:
            ax.grid(visible=True)
            ax.axhline(0, color="k")

    for fig, name in zip(figures, fignum.keys()):
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                project["paths"]["output_path"], f"Movements_{name}.png"
            )
        )


def add_profile(
    ax: maxs.Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    ii: int,
    xoffset: float = 0,
):
    x = np.round(xdata, -2)
    y = ydata
    xq = np.unique(x)
    yq = np.array([np.nanmedian(y[x == ux]) for ux in xq])
    yqsu = np.array([np.nanpercentile(y[x == ux], 95) for ux in xq])
    yqsl = np.array([np.nanpercentile(y[x == ux], 5) for ux in xq])
    if not xoffset:
        xoffset = np.min(xq)
    xq = xq - xoffset
    ax.errorbar(
        xq,
        yq,
        yerr=[np.abs(yq - yqsl), np.abs(yq - yqsu)],
        marker="s",
        color=f"C{ii}",
        linestyle="None",
        capsize=2,
    )


if __name__ == "__main__":
    main()
