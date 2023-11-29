""" Impact of construction activity on PSI motion """
import os
import warnings

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

import u4py.analysis.inversion as u4invert
import u4py.analysis.processing as u4proc
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects


def main():
    # Load Data
    warnings.filterwarnings("ignore")
    project = u4projects.get_project(
        required=[
            "base_path",
            "psivert_path",
            "subsubregions_path",
            "output_path",
            "processing_path",
            "places_path",
        ],
        interactive=False,
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
        print("Inverting", sel_shape.Name)
        sel_points, sel_fpath = u4files.get_region_points(
            sel_shape,
            sel_name,
            project["paths"]["psivert_path"],
            # overwrite=True,
            crs=selection_shapes.crs,
        )
        sel_data = u4files.load_data_from_points(
            project["paths"]["psivert_path"], sel_points
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
            # keys=["CRUMSTADT", "HAHN flach", "ALLMENDFELD (alt)"],
            # labels=["Crumstadt", "Hahn\n(shallow)", "Allmendfeld\n(old)"],
        )

        shape_colors = ["k"] * sel_shape.size
        shape_colors[ii] = "C0"
        u4ax.add_shapefile(
            project["paths"]["subsubregions_path"],
            ax=axes[0],
            facecolor="none",
            edgecolor=shape_colors,
            crs=selection_shapes.crs,
        )

        # Timeseries Fit and Residuals
        t_EX = ext_times[sel_name]
        results = u4proc.invert_psi_dict(
            sel_data,
            save_path=os.path.join(
                project["paths"]["processing_path"],
                f"{shp_fname}_{sel_name}.pkl",
            ),
            # num_coeffs=3,
            t_EX=t_EX,
            # overwrite=True,
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
            # zoom=15,
            source=contextily.providers.OpenStreetMap.Mapnik,
        )
        fig.tight_layout()
        print("Saving Plots...")
        fig.savefig(
            os.path.join(
                project["paths"]["output_path"],
                f"{shp_fname}_{sel_name}_Timeseries.pdf",
            )
        )
        fig.savefig(
            os.path.join(
                project["paths"]["output_path"],
                f"{shp_fname}_{sel_name}_Timeseries",
            )
        )
        print(f"Results for {sel_shape.Name}")
        print(
            u4invert.print_inversion_results(
                results["U"]["ori_inversion_results"],
                results["U"]["parameters_list"],
            )
        )
        plt.close(fig)


def plot_map(ax, data_region, crs):
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
