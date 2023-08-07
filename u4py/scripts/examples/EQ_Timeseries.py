import os

import contextily
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt

import u4py.addons.seismo as u4seismo
import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4fmt
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        required=[
            "base_path",
            "bld_path",
            "psivert_path",
            "ext_path",
            "processing_path",
            "output_path",
        ],
        interactive=False,
    )
    file_path = os.path.join(
        project["paths"]["ext_path"],
        "Erdbeben",
        "Erdbebenkatalog_HED_20230727.xlsx",
    )
    eq_gdf = u4seismo.get_prepared_eq_catalogue(
        file_path, project["paths"]["bld_path"]
    )
    buffer = 1000
    points = u4spatial._select_points_point(
        eq_gdf, buffer, project["paths"]["psivert_path"], split_points=True
    )

    # MAP
    fig = plt.figure(figsize=(18, 7), dpi=150)
    grid = gs.GridSpec(ncols=3, nrows=len(points))
    axes = [fig.add_subplot(grid[:, 0])]
    create_map(eq_gdf, points, buffer, ax=axes[0])
    for ii, pnt in enumerate(points):
        axes.append(fig.add_subplot(grid[ii, 1:]))
        data = u4files.load_data_from_points(
            project["paths"]["psivert_path"], pnt
        )
        eqid = list(eq_gdf.ID)[ii]
        eqloc = list(eq_gdf.LOKATION)[ii]
        eqdep = list(eq_gdf.HERDTIEFE)[ii]
        eqmag = list(eq_gdf.LOKALMAGNITUDE)[ii]
        eqtime = f"{list(eq_gdf.DATETIME)[ii].isoformat()}".replace(":", "-")

        inversion_path = os.path.join(
            project["paths"]["processing_path"], f"{eqid}_{eqloc}_{eqtime}.pkl"
        )
        results = u4proc.invert_psi_dict(data, save_path=inversion_path)
        u4ax.plot_timeseries_fit(ax=axes[ii + 1], results=results)
        axes[ii + 1].annotate(
            f"$M_L${eqmag}, {eqloc} ({eqdep} km)",
            (0.97, 0.97),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
            fontweight="bold",
        )
        axes[ii + 1].axvline(
            list(eq_gdf.DATETIME)[ii], color=f"C{ii}", linewidth=4
        )
    fig.tight_layout()
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "EQ_Timeseries.pdf")
    )
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "EQ_Timeseries.png")
    )


def create_map(eq_gdf, points, buffer, ax):
    eq_gdf.plot(ax=ax, color="k", marker="*", zorder=3)
    eq_gdf.buffer(buffer).boundary.plot(ax=ax, zorder=2, color="k")
    for pnt in points:
        pnt.plot(ax=ax)
    contextily.add_basemap(
        ax,
        crs=eq_gdf.crs.to_string(),
        # zoom=15,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    u4fmt.map_style(ax=ax)


if __name__ == "__main__":
    main()
