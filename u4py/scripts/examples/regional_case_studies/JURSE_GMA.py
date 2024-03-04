import os
from pathlib import Path

import contextily
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import u4py.analysis.spatial as u4spatial
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4pltfmt
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\JURSE_GMA.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "output_path",
            "processing_path",
            "psi_path",
            "places_path",
        ],
        interactive=False,
    )
    # Limits
    vm_gma = 4
    vm_mv = 3

    # Paths
    gma_fpath = os.path.join(
        project["paths"]["processing_path"],
        "GMA_results",
        # "GMA_results_cell=100_minmean=2_maxvar=5.tif",
        "GMA_results_cell=250_minmean=1.5_maxvar=1.tif",
    )
    # ada_fpath = os.path.join(
    #     project["paths"]["base_path"],
    #     "Results_ADA_tools",
    #     "merged_ada_results",
    #     "merged_vertikal_adas.shp",
    # )
    psi_fpath = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
    )

    # Regions
    regions = {
        "ffm": (460000, 5540000, 479000, 5555000),
        "crum": (462000, 5514000, 470000, 5520400),
    }
    data_ffm = u4psi.get_region_data(
        gp.GeoDataFrame(
            geometry=[u4spatial.bounds_to_polygon(regions["ffm"])],
            crs="EPSG:32632",
        ),
        "ffm_gma",
        psi_fpath,
        overwrite=True,
    )

    data_crum = u4psi.get_region_data(
        gp.GeoDataFrame(
            geometry=[u4spatial.bounds_to_polygon(regions["crum"])],
            crs="EPSG:32632",
        ),
        "crum_gma",
        psi_fpath,
    )

    station_list = [
        "CRUMSTADT",
        "HAHN flach",
        "ALLMENDFELD (alt)",
        "GODDELAU",
    ]
    label_list = [
        "Crumstadt",
        "Hahn (shallow)",
        "Allmendfeld (old)",
        "Goddelau",
    ]

    fig, axes = plt.subplots(figsize=(14, 11), dpi=300, ncols=2, nrows=2)

    u4ax.add_tile(
        gma_fpath,
        ax=axes[1, 0],
        cmap="Oranges",
        zorder=2,
        vm=(-1, vm_gma),
        colorbar={
            "label": "GMA Result",
            "shrink": 0.7,
            "orientation": "horizontal",
            "extend": "max",
            "pad": 0.15,
        },
    )
    u4ax.add_tile(
        gma_fpath,
        ax=axes[1, 1],
        cmap="Oranges",
        zorder=2,
        vm=(-1, vm_gma),
        colorbar={
            "label": "GMA Result",
            "shrink": 0.7,
            "orientation": "horizontal",
            "extend": "max",
            "pad": 0.15,
        },
    )
    u4spatial.xy_data_to_gdf(
        data_ffm["x"], data_ffm["y"], data_ffm["mean_vel"], crs="EPSG:32632"
    ).plot(
        ax=axes[0, 0],
        column="data",
        cmap="RdYlBu",
        vmin=-vm_mv,
        vmax=vm_mv,
        zorder=2,
        markersize=3,
        legend=True,
        legend_kwds={
            "label": "Mean Vertical Velocity (mm/a)",
            "orientation": "horizontal",
            "shrink": 0.7,
            "extend": "both",
            "pad": 0.15,
        },
    )
    u4spatial.xy_data_to_gdf(
        data_crum["x"], data_crum["y"], data_crum["mean_vel"], crs="EPSG:32632"
    ).plot(
        ax=axes[0, 1],
        column="data",
        cmap="RdYlBu",
        vmin=-vm_mv,
        vmax=vm_mv,
        zorder=2,
        markersize=3,
        legend=True,
        legend_kwds={
            "label": "Mean Vertical Velocity (mm/a)",
            "orientation": "horizontal",
            "shrink": 0.7,
            "extend": "both",
            "pad": 0.15,
        },
    )

    plot_map_ffm(axes[0, 0], project)

    # The map
    plot_map_crumstadt(
        axes[0, 1],
        "EPSG:32632",
        project["paths"]["places_path"],
        station_list,
        label_list,
        psi_fpath,
    )

    axes[0, 0].set_xlim(regions["ffm"][0], regions["ffm"][2])
    axes[0, 0].set_ylim(regions["ffm"][1], regions["ffm"][3])
    axes[1, 0].set_xlim(regions["ffm"][0], regions["ffm"][2])
    axes[1, 0].set_ylim(regions["ffm"][1], regions["ffm"][3])

    axes[0, 1].set_xlim(regions["crum"][0], regions["crum"][2])
    axes[0, 1].set_ylim(regions["crum"][1], regions["crum"][3])
    axes[1, 1].set_xlim(regions["crum"][0], regions["crum"][2])
    axes[1, 1].set_ylim(regions["crum"][1], regions["crum"][3])
    for ax in axes.flat:
        u4pltfmt.map_style(ax=ax, divisor=2000, crs="EPSG:32632")
        u4ax.add_basemap(
            ax=ax,
            crs="EPSG:32632",
            source=contextily.providers.CartoDB.Voyager,
        )

    u4pltfmt.enumerate_axes(fig, ignore=[4, 5, 6, 7])
    # fig.tight_layout()
    fig.subplots_adjust(
        left=0.1, right=0.8, top=0.95, bottom=0.05, hspace=0.025
    )
    fig.savefig(os.path.join(project["paths"]["output_path"], "JURSE_GMA.png"))
    fig.savefig(os.path.join(project["paths"]["output_path"], "JURSE_GMA.pdf"))
    plt.close(fig)


def plot_map_ffm(ax, project):

    u4ax.add_shapefile(
        ax=ax,
        shp_path=os.path.join(
            project["paths"]["places_path"],
            "OSM_shapes",
            "gis_osm_railways_free_1.shp",
        ),
        color="gray",
        linewidth=0.5,
    )
    u4ax.add_shapefile(
        ax=ax,
        shp_path=os.path.join(
            project["paths"]["places_path"], "Highrise_buildings.shp"
        ),
        marker=".",
        # markersize=10,
        color="k",
        zorder=15,
    )
    u4ax.add_shapefile(
        ax=ax,
        shp_path=os.path.join(
            project["paths"]["places_path"], "FFM_regions.shp"
        ),
        fc="None",
        ec=["C2", "C3", "C1", "C0"],
        linewidth=2,
        zorder=14,
    )


def plot_map_crumstadt(
    ax, crs, places_path, station_list, label_list, psi_path
):
    """Adds a map with some annotations to the given axis.

    :param ax: The axis to add the plot to.
    :type ax: Axes
    :param data_region: The PSI-data for the region.
    :type data_region: dict
    :param crs: The coordinate system of the map.
    :type crs: str
    :param places_path: The path to additional shapefiles that are to be added.
    :type places_path: os.PathLike
    """
    _, region_well = u4psi.get_point_data(
        (464350, 5516100), 500, "OilWell", psi_path
    )
    query = {
        "address": "Crumstadt",
        "tags": {
            "landuse": ["residential", "industrial"],
            "amenity": "hospital",
        },
    }
    _, region_osm = u4psi.get_osm_data(query, psi_path)
    # Map
    u4pltfmt.add_map_label("Crumstadt", (465500, 5518000), ax=ax)
    u4pltfmt.add_map_label("Hahn", (468700, 5516000), ax=ax)
    u4pltfmt.add_map_label("Gas Storage", (464500, 5516100), ax=ax)

    ax.plot(
        464350,
        5516100,
        color="k",
        marker="^",
        linewidth=0,
        markersize=8,
        zorder=4,
    )

    u4ax.add_shapefile(
        os.path.join(places_path, "Gebaudeschaeden.shp"),
        crs=crs,
        ax=ax,
        color="k",
        marker="x",
        markersize=20,
        zorder=4,
    )

    u4ax.add_shapefile(
        os.path.join(places_path, "Tiefenlinie_Top_Sand_7.shp"),
        ax=ax,
        column="Z",
        facecolor="none",
        zorder=3,
    )
    u4ax.add_shapefile(
        os.path.join(places_path, "Gas_Störungen.shp"),
        ax=ax,
        color="k",
        linestyle=":",
        zorder=3,
    )

    u4ax.add_shapefile(
        ax=ax,
        shp_path=os.path.join(
            places_path,
            "OSM_shapes",
            "gis_osm_railways_free_1.shp",
        ),
        color="gray",
        linewidth=0.5,
        zorder=4,
    )

    region_osm.plot(
        ax=ax, fc="None", ec="C1", linewidth=2, zorder=4
    )  # , linestyle="--")
    region_well.plot(
        ax=ax, fc="None", ec="C2", linewidth=2, zorder=4
    )  # , linestyle="--")

    u4ax.add_shapefile(
        os.path.join(places_path, "GW_Stations.shp"),
        ax=ax,
        marker=u4pltfmt.drop_shape(),
        color="b",
        markersize=50,
        zorder=4,
        keys=station_list,
        labels=label_list,
    )
    u4ax.add_shapefile(
        os.path.join(places_path, "WeatherStationCoordinates_HLNUG.shp"),
        ax=ax,
        marker="d",
        color="C3",
        markersize=30,
        zorder=4,
        keys=["Riedstadt", "Darmstadt", "Mörfelden", "Raunheim"],
        labels=["Riedstadt", "Darmstadt", "Mörfelden", "Raunheim"],
    )

    virid = mpl.colormaps.get_cmap("viridis")(np.linspace(0, 1, 6))
    handles = [
        # Line2D(
        #     [], [], linestyle="None", color="k", marker=".", label="PSI Data"
        # ),
        # Line2D(
        #     [],
        #     [],
        #     linewidth=2,
        #     color="C1",
        #     # linestyle="--",
        #     label="Crumstadt PS Area",
        # ),
        # Line2D(
        #     [],
        #     [],
        #     linewidth=2,
        #     color="C2",
        #     # linestyle="--",
        #     label="Gas Storage PS Area",
        # ),
        Line2D([], [], color=virid[0], label="400 m Reservoir Depth"),
        Line2D([], [], color=virid[1], label="410 m"),
        Line2D([], [], color=virid[2], label="420 m"),
        Line2D([], [], color=virid[3], label="430 m"),
        Line2D([], [], color=virid[4], label="440 m"),
        Line2D([], [], color=virid[5], label="450 m"),
        Line2D([], [], color="k", linewidth=2, label="Fault", linestyle=":"),
        Line2D([], [], color="gray", label="Railways"),
        Line2D(
            [],
            [],
            color="k",
            marker="^",
            markersize=8,
            linestyle="None",
            label="Location Gas Storage",
        ),
        Line2D(
            [],
            [],
            marker=u4pltfmt.drop_shape(),
            color="b",
            markersize=10,
            linestyle="None",
            label="Groundwater Wells",
        ),
        Line2D(
            [],
            [],
            marker="d",
            color="C3",
            markersize=8,
            linestyle="None",
            label="Weather Stations",
        ),
        Line2D(
            [],
            [],
            marker="x",
            color="k",
            linestyle="None",
            label="Damaged Structures",
        ),
        Line2D(
            [],
            [],
            marker=".",
            linestyle="None",
            color="k",
            label="Highrise Buildings",
        ),
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
    )


if __name__ == "__main__":
    main()
