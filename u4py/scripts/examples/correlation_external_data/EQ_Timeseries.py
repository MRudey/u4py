import os

import contextily
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt

import u4py.addons.seismo as u4seismo
import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4fmt
import u4py.utils.projects as u4proj


def main():
    overwrite = True
    project = u4proj.get_project(
        required=[
            "base_path",
            "bld_path",
            "psi_path",
            "ext_path",
            "processing_path",
            "output_path",
        ],
        interactive=False,
    )
    psi_fpath = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
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

    # MAP
    fig = plt.figure(figsize=(18, 7), dpi=150)
    grid = gs.GridSpec(ncols=3, nrows=len(eq_gdf))
    axes = [fig.add_subplot(grid[:, 0])]
    create_map(eq_gdf, buffer, ax=axes[0])
    for ii, pnt in enumerate(eq_gdf.geometry):
        axes.append(fig.add_subplot(grid[ii, 1:]))
        data = u4psi.get_point_data(
            pnt,
            radius=buffer,
            region_name=eq_gdf.LOKATION.iloc[ii],
            source_fpath=psi_fpath,
            overwrite=overwrite,
        )
        u4spatial.xy_data_to_gdf(data["x"], data["y"], crs=eq_gdf.crs).plot(
            ax=axes[0], color=f"C{ii}"
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


def create_map(eq_gdf, buffer, ax):
    eq_gdf.plot(ax=ax, color="k", marker="*", zorder=3)
    eq_gdf.buffer(buffer).boundary.plot(ax=ax, zorder=2, color="k")
    contextily.add_basemap(
        ax,
        crs=eq_gdf.crs.to_string(),
        # zoom=15,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    u4fmt.map_style(ax=ax)


if __name__ == "__main__":
    main()
