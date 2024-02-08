"""
Impact of construction activity on PSI movement in the city centre of
Frankfurt a.M.

Several regions were identified and outlined as shapes in a shapefile of
`subsubregions`. For each of these a separate plot is generated including a map
of all PSs, timeseries and residuals. This simplifies the identification of
transient movements for inversion.
"""

import os
import warnings

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import u4py.analysis.processing as u4proc
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.plotting.preparation as u4plotprep
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4projects

u4config.start_logger()


def main():
    # Load Data
    warnings.filterwarnings("ignore")
    overwrite = True
    project = u4projects.get_project(
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
            (u4convert.get_datetime(2016), -25),
            (u4convert.get_datetime(2019), -22),
        ],
        "Outer_Subsidence": [],
        "Uplift": [
            (u4convert.get_datetime(2016.25), -7),
            (u4convert.get_datetime(2018.5), -7.5),
        ],
    }
    fit_nums = {
        "Inner_City": 2,
        "Inner_Subsidence": 1,
        "Outer_Subsidence": 2,
        "Uplift": 1,
    }

    for ii, sel_shape in selection_shapes.iterrows():
        sel_name = (sel_shape.Name).replace(" ", "_")
        if sel_name == "FFM_Hoechst":
            break
        print("=============================")
        print("Inverting", sel_shape.Name)
        sel_data = u4psi.get_region_data(
            sel_shape,
            sel_name,
            psi_fpath,
            overwrite=overwrite,
            crs=selection_shapes.crs,
        )

        fig = plt.figure(figsize=(16 * 0.75, 10 * 0.75), dpi=150)
        grid = gs.GridSpec(ncols=2, nrows=2)
        axes = [
            fig.add_subplot(grid[:, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]),
        ]

        # Map
        plot_map(axes[0], sel_data, selection_shapes.crs)
        u4ax.add_shapefile(
            os.path.join(project["paths"]["places_path"], "GW_Stations.shp"),
            ax=axes[0],
            marker=u4plotfmt.drop_shape(),
            color="b",
            markersize=50,
            zorder=3,
            label="Groundwater Wells",
        )

        u4ax.add_shapefile(
            project["paths"]["subsubregions_path"],
            ax=axes[0],
            facecolor="none",
            edgecolor="k",
            crs=selection_shapes.crs,
        )
        gp.GeoSeries(sel_shape["geometry"], crs=selection_shapes.crs).plot(
            ax=axes[0], facecolor="none", edgecolor="C0"
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
        )
        u4ax.plot_timeseries_fit(
            ax=axes[1],
            results=results,
            fit_num=fit_nums[sel_name],
            annotate=False,
        )
        u4ax.plot_fit_residuals(
            ax=axes[2],
            data=sel_data,
            fit_data=results,
            fit_num=fit_nums[sel_name],
        )

        # Formatting
        axes[2].sharex(axes[1])
        axes[1].set_ylabel("Vertical Displacement (mm)")
        axes[2].set_ylabel("Residual Displacement (mm)")
        axes[2].set_xlabel("Year")
        fig.suptitle(sel_shape.Name)

        ylims = axes[1].get_ylim()
        ylow = ylims[0] + 0.2 * (ylims[1] - ylims[0])
        if t_EX:
            parlist = results["U"]["parameters_list"]
            vals = [
                results["U"]["ori_inversion_results"][ii]
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
                if ylow < axes[1].get_ylim()[0]:
                    axes[1].set_ylim(ylow, ylims[1])
                axes[1].add_patch(arrow)

            for annpos, vv in zip(annot_pos[sel_name], vals):
                if vv > 0:
                    col = "C0"
                else:
                    col = "C3"
                axes[1].annotate(
                    f"{vv:.2f} mm",
                    annpos,
                    horizontalalignment="center",
                    color=col,
                )
        axes[0].set_xlim(470500, 479000)
        axes[0].set_ylim(5548000, 5555000)
        contextily.add_basemap(
            axes[0],
            crs=selection_shapes.crs,
            source=contextily.providers.OpenStreetMap.Mapnik,
        )
        fig.tight_layout()
        print("Saving Plots...")
        output_path = os.path.join(
            project["paths"]["output_path"], "U5_Frankfurt"
        )
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(
                output_path,
                f"{shp_fname}_{sel_name}_Timeseries.pdf",
            )
        )
        fig.savefig(
            os.path.join(
                output_path,
                f"{shp_fname}_{sel_name}_Timeseries",
            )
        )
        print("----------------------------------")
        u4plotprep.print_inversion_results_for_publications(results)
        plt.close(fig)


def plot_map(ax: Axes, data_region: dict, crs: str):
    # Map
    u4ax.plot_region_trend(data_region, ax=ax)
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    ax.annotate(
        f"#PSI = {len(data_region['x'])}", (0, -0.1), xycoords="axes fraction"
    )
    u4plotfmt.map_style(ax, divisor=1000, crs=crs)


if __name__ == "__main__":
    main()
