"""Impact of construction activity on PSI motion in Frankfurt a.M."""

import os
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gp
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

import u4py.analysis.processing as u4proc
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.plotting.preparation as u4plotprep
import u4py.utils.config as u4config
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    # Load Data
    warnings.filterwarnings("ignore")
    overwrite = False
    project = u4projects.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Examples\U5_Frankfurt_and_JURSE_FFM_GPKG.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "psi_path",
            "subsubregions_path",
            "output_path",
            "processing_path",
            "places_path",
        ],
        interactive=False,
    )
    psi_fpath = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
    )
    shp_fname = os.path.splitext(
        os.path.split(project["paths"]["subsubregions_path"])[-1]
    )[0]
    selection_shapes = gp.read_file(project["paths"]["subsubregions_path"])

    # Set Water Extraction times:
    ext_times = {
        "Inner_City": [],
        "Inner_Subsidence": [
            (2015, 2017),
            (2018.5, 2020),
        ],
        "Outer_Subsidence": [],
        "Uplift": [(2016.75, 2017.5), (2017.5, 2018)],
    }

    fit_nums = {
        "Inner_City": 2,
        "Inner_Subsidence": 2,
        "Outer_Subsidence": 2,
        "Uplift": 1,
    }

    ax_nums = {
        "Inner_City": 0,
        "Inner_Subsidence": 2,
        "Outer_Subsidence": 1,
        "Uplift": 3,
    }

    fig, axes = plt.subplots(
        figsize=(7, 10), dpi=300, nrows=4, sharex=True, sharey=True
    )

    print("========================")
    for ii, sel_shape in selection_shapes.iterrows():
        sel_name = (sel_shape.Name).replace(" ", "_")
        ax_num = ax_nums[sel_name]
        print("Inverting", sel_shape.Name)
        sel_data = u4psi.get_region_data(
            sel_shape,
            sel_name,
            psi_fpath,
            overwrite=overwrite,
            crs=selection_shapes.crs,
        )

        # Timeseries Fit and Residuals
        t_EX = ext_times[sel_name]
        results = u4proc.get_psi_dict_inversion(
            sel_data,
            save_path=os.path.join(
                project["paths"]["processing_path"],
                f"{shp_fname}_{sel_name}.pkl",
            ),
            t_EX=t_EX,
            overwrite=overwrite,
        )

        u4plotprep.print_inversion_results_for_publications(results)

        u4ax.plot_timeseries_fit(
            ax=axes[ax_num],
            results=results,
            fit_num=fit_nums[sel_name],
            annotate=False,
            color=f"C{ax_num}",
            color_fit="k",
        )

        axes[ax_num].annotate(
            sel_shape.Name,
            (0.99, 0.01),
            xycoords="axes fraction",
            fontweight="bold",
            horizontalalignment="right",
        )
        print("========================")

    # axes[0].set_ylim(-35, 15)
    axes[0].set_xlim(
        datetime(2015, 1, 1),
        # datetime(2021, 1, 1),
    )
    for ax in axes:
        ax.set_ylabel("d$_{vert}$ (mm)")
    axes[-1].set_xlabel("Year")
    axes[0].legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                marker=".",
                color="C0",
                linewidth=0,
                label="Median",
            ),
            mpatch.Patch(fc="C0", alpha=0.5, label="95% range"),
            mpatch.Patch(fc="C0", alpha=0.75, label="68% range"),
            mlines.Line2D([], [], color="k", label="Fit"),
        ],
        loc="lower left",
        # ncols=4,
        markerscale=0.75,
        fontsize="small",
    )
    u4plotfmt.enumerate_axes(fig)
    fig.tight_layout()
    print("Saving Plots...")
    output_path = os.path.join(project["paths"]["output_path"], "U5_Frankfurt")
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(
        os.path.join(
            output_path,
            f"JURSE_FFM_Timeseries.pdf",
        )
    )
    fig.savefig(
        os.path.join(
            output_path,
            f"JURSE_FFM_Timeseries",
        )
    )


if __name__ == "__main__":
    main()
