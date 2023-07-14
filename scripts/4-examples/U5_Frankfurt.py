""" Impact of construction activity on PSI motion """
import os
import warnings

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import scipy.stats as spstats
from tqdm import tqdm

import u4py.analysis.processing as u4proc
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.files as u4files
import u4py.utils.projects as u4projects


def main():
    warnings.filterwarnings("ignore")

    project = u4projects.get_project(
        required=[
            "base_path",
            "psivert_path",
            "subsubregions_path",
            "output_path",
            "processing_path",
        ],
        interactive=False,
    )
    shp_fname = os.path.splitext(
        os.path.split(project["paths"]["subsubregions_path"])[-1]
    )[0]
    selection_shapes = gp.read_file(project["paths"]["subsubregions_path"])
    for ii, sel_shape in tqdm(
        selection_shapes.iterrows(), total=len(selection_shapes)
    ):
        sel_name = (sel_shape.Name).replace(" ", "_")
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
        grid = gs.GridSpec(ncols=2, nrows=3)
        axes = [
            fig.add_subplot(grid[:, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, 1]),
        ]

        # Map
        plot_map(axes[0], sel_data)

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
        results = u4proc.invert_psi_dict(
            sel_data,
            save_path=os.path.join(
                project["paths"]["processing_path"],
                f"{shp_fname}_{sel_name}.pkl",
            ),
        )
        u4ax.plot_timeseries_fit(ax=axes[1], results=results)
        u4ax.plot_fit_residuals(ax=axes[2], data=sel_data, fit_data=results)

        # Residuals statistics
        u4ax.plot_pdf(ax=axes[3], y=results["U"]["y_fit_2_err"])

        # Formatting
        axes[0].legend(loc="upper right")
        axes[2].sharex(axes[1])
        for ax in axes[1:-1]:
            ax.set_ylabel("Vertical Displacement (mm)")
        axes[2].set_xlabel("Year")
        axes[3].set_xlabel("Residual Displacement (mm)")
        axes[3].set_ylabel("PDF (1)")
        fig.suptitle(sel_shape.Name)
        contextily.add_basemap(
            axes[0],
            crs=selection_shapes.crs,
            # zoom=15,
            source=contextily.providers.OpenStreetMap.Mapnik,
        )
        fig.tight_layout()
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


def plot_map(ax, data_region):
    # Map
    u4ax.plot_region_trend(data_region, ax=ax)
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    u4plotfmt.map_style(ax, divisor=1000)


if __name__ == "__main__":
    main()
