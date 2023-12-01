""" Impact of construction activity on PSI motion in Frankfurt a.M."""
import os
import warnings
from pathlib import Path

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

import u4py.analysis.processing as u4proc
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.plotting.preparation as u4plotprep
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    # Load Data
    warnings.filterwarnings("ignore")
    overwrite = True
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

    annot_pos = {
        "Inner_City": [],
        "Inner_Subsidence": [
            (u4convert.get_datetime(2016), -15),
            (u4convert.get_datetime(2019), -25),
        ],
        "Outer_Subsidence": [],
        "Uplift": [
            (u4convert.get_datetime(2016.25), -10),
            (u4convert.get_datetime(2018.5), -10),
        ],
    }
    fit_nums = {
        "Inner_City": 2,
        "Inner_Subsidence": 1,
        "Outer_Subsidence": 2,
        "Uplift": 1,
    }

    ax_nums = {
        "Inner_City": 0,
        "Inner_Subsidence": 2,
        "Outer_Subsidence": 1,
        "Uplift": 3,
    }

    fig = plt.figure(figsize=(16 * 0.75, 10 * 0.75), dpi=150)
    grid = gs.GridSpec(ncols=2, nrows=4)
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, 1]),
    ]
    print("========================")
    for ii, sel_shape in selection_shapes.iterrows():
        sel_name = (sel_shape.Name).replace(" ", "_")
        if sel_name == "FFM_Hoechst":
            break
        ax_num = ax_nums[sel_name] + 1
        print("Inverting", sel_shape.Name)
        sel_data = u4files.get_region_data(
            sel_shape,
            sel_name,
            psi_fpath,
            overwrite=overwrite,
            crs=selection_shapes.crs,
        )

        if sel_name == "Inner_City":
            plot_map(axes[0], sel_data, selection_shapes.crs)
        gp.GeoSeries(sel_shape["geometry"]).plot(
            ax=axes[0],
            facecolor="None",
            edgecolor=f"C{ax_num-1}",
            zorder=3,
            linewidth=3,
        )
        # Timeseries Fit and Residuals
        t_EX = ext_times[sel_name]
        results = u4proc.invert_psi_dict(
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
            color=f"C{ax_num-1}",
            color_fit="k",
        )

        axes[ax_num].annotate(
            sel_shape.Name,
            (0.99, 0.9),
            xycoords="axes fraction",
            horizontalalignment="right",
        )

        ylims = axes[ax_num].get_ylim()
        ylow = ylims[0] + 0.2 * (ylims[1] - ylims[0])
        if t_EX:
            parlist = results["U"]["parameters_list"]
            vals = [
                results["U"]["inversion_results"][ii]
                for ii in range(len(parlist))
                if "water extraction" in parlist[ii]
            ]

            for vv, (ts, te) in zip(vals, t_EX):
                if vv > 0:
                    col = "C0"
                else:
                    col = "C3"

                arrow = mpatch.FancyArrowPatch(
                    (u4convert.get_datetime(ts), ylow),
                    (u4convert.get_datetime(te), ylow + vv),
                    fc=col,
                    mutation_scale=25,
                )
                ylow = ylow + vv
                if ylow < axes[ax_num].get_ylim()[0]:
                    axes[ax_num].set_ylim(ylow, ylims[1])
                axes[ax_num].add_patch(arrow)

            for annpos, vv in zip(annot_pos[sel_name], vals):
                if vv > 0:
                    col = "C0"
                else:
                    col = "C3"
                axes[ax_num].annotate(
                    f"{vv:.2f} mm",
                    annpos,
                    horizontalalignment="center",
                    color=col,
                )
        print("========================")
    print("Formatting Plot")
    # Formatting
    axes[0].set_xlim(470500, 479000)
    axes[0].set_ylim(5548000, 5555000)
    contextily.add_basemap(
        axes[0],
        crs=selection_shapes.crs,
        # zoom=17,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    axes[1].set_ylim(-30, 10)
    for ax in axes[1:]:
        ax.set_ylabel("d$_{vert}$ (mm)")
        ax.sharex(axes[1])
        ax.sharey(axes[1])
    # axes[-1].set_yticklabels(rotation=0)
    axes[-1].set_xlabel("Year")
    axes[1].legend(
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
        loc="lower center",
        ncols=4,
        markerscale=0.75,
        fontsize="small",
    )
    u4plotfmt.enumerate_axes(fig, ignore=[5])
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


def plot_map(ax, data_region, crs):
    # Map
    u4ax.plot_region_trend(data_region, ax=ax)
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    ax.annotate(
        f"#PSI = {len(data_region['x'])}", (0, -0.1), xycoords="axes fraction"
    )
    u4plotfmt.map_style(ax, divisor=2000, crs=crs)


if __name__ == "__main__":
    main()
