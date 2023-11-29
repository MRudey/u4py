""" Shows possible correlation of surface motion with gas storage activity """

import os

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely import Polygon

import u4py.addons.gas_storage as u4gas
import u4py.analysis.processing as u4proc
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    # Paths
    project = u4proj.get_project(
        required=[
            "base_path",
            "psivert_path",
            "results_path",
            "ext_path",
            "places_path",
            "output_path",
            "processing_path",
        ]
    )

    points = u4files.get_point_points(
        (464350, 5516100), 500, "OilWell", project["paths"]["psivert_path"]
    )

    # Load Data
    data_well = u4files.load_data_from_points(
        project["paths"]["psivert_path"], points
    )
    results = u4proc.invert_psi_dict(
        data_well,
        save_path=os.path.join(
            project["paths"]["processing_path"], "GasStorage.pkl"
        ),
    )

    date_gas, _, level_gas = u4gas.load_gas_data(
        os.path.join(
            project["paths"]["ext_path"], "Inventory Turnover Data_23.txt"
        )
    )
    _, crs = get_data_within_osm_query(project["paths"]["psivert_path"])
    data_region, _ = get_data_within_region(
        project["paths"]["psivert_path"], crs=crs
    )

    # fig, axes = plt.subplots(figsize=(10, 8), sharex=True)
    fig = plt.figure(figsize=(16 * 0.75, 10 * 0.75), dpi=150)
    grid = gs.GridSpec(ncols=2, nrows=3)
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 1]),
    ]

    # The map
    plot_map(
        axes[0],
        data_region,
        crs=crs,
        places_path=project["paths"]["places_path"],
    )
    u4ax.plot_timeseries_fit(ax=axes[1], results=results)
    # u4ax.plot_timeseries(
    #     data_well["time"], data_well["timeseries"], ax=axes[1], color="C0"
    # )
    # u4ax.plot_inversion_fit(
    #     data_well["time"], inversion_data[0][2], ax=axes[1]
    # )

    u4ax.plot_fit_residuals(
        data_well, results["U"]["inversion_results"], ax=axes[2]
    )

    axes[3].plot(
        date_gas, level_gas, color="C2", linewidth=2, label="Fill level"
    )
    u4plotfmt.add_copyright("Source: MND Energies, online", ax=axes[3])

    # Formatting
    axes[0].legend(loc="upper right")
    axes[1].legend(ncols=3, loc="best")
    axes[2].legend(loc="best")
    axes[3].legend(loc="lower right")
    axes[2].sharex(axes[1])
    axes[3].sharex(axes[1])
    for ax in axes[1:-1]:
        ax.set_ylabel("Vertical Displacement (mm)")
    axes[3].set_ylabel("Fill level of gas storage (%)")
    axes[3].set_xlabel("Year")

    # plt.show()
    contextily.add_basemap(
        axes[0],
        crs=crs,
        # zoom=15,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            project["paths"]["output_path"], "Riedstadt_Timeseries.pdf"
        )
    )
    fig.savefig(
        os.path.join(
            project["paths"]["output_path"], "Riedstadt_Timeseries.svg"
        )
    )
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "Riedstadt_Timeseries")
    )


def add_gas_geology(places_path: os.PathLike, ax: Axes):
    u4ax.add_shapefile(
        os.path.join(places_path, "Tiefenlinie_Top_Sand_7.shp"),
        ax=ax,
        column="Z",
        facecolor="none",
        zorder=1,
    )
    u4ax.add_shapefile(
        os.path.join(places_path, "Gas_Störungen.shp"),
        ax=ax,
        color="k",
        label="Faults",
        zorder=1,
    )


def plot_map(ax, data_region, crs, places_path):
    # Map
    u4ax.plot_region_trend(data_region, ax=ax)
    u4plotfmt.add_map_label("Crumstadt", (465500, 5518000), ax=ax)
    u4plotfmt.add_map_label("Hahn", (468700, 5516000), ax=ax)
    u4plotfmt.add_map_label("Gas Storage", (464500, 5516100), ax=ax)

    ax.plot(
        464350,
        5516100,
        color="k",
        marker="^",
        linewidth=0,
        markersize=10,
    )

    u4ax.add_shapefile(
        os.path.join(places_path, "Gebaudeschaeden.shp"),
        crs=crs,
        ax=ax,
        color="k",
        # markersize=,
        label="Known Building Damage",
        marker="x",
        markersize=8,
    )

    u4ax.add_shapefile(
        os.path.join(places_path, "Tiefenlinie_Top_Sand_7.shp"),
        ax=ax,
        column="Z",
        facecolor="none",
        zorder=1,
    )
    u4ax.add_shapefile(
        os.path.join(places_path, "Gas_Störungen.shp"),
        ax=ax,
        color="k",
        label="Faults",
        zorder=1,
    )

    ax.set_xlim(463200, 469500)
    ax.set_ylim(5514800, 5520400)
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    u4plotfmt.map_style(ax, divisor=2000)


def get_data_within_osm_query(h5path):
    # Define osm query
    query = {
        "address": "Crumstadt",
        "tags": {
            "landuse": ["residential", "industrial"],
            "amenity": "hospital",
        },
    }
    # Get points from file, if not available creates new file
    points = u4files.get_osm_points(query, h5path)
    data = u4files.load_data_from_points(h5path, points)
    return data, points.crs


def get_data_within_region(h5path, crs):
    region = gp.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (463200, 5514800),
                        (463200, 5520400),
                        (469500, 5520400),
                        (469500, 5514800),
                    ]
                )
            ]
        },
        crs=crs,
    )
    points, _ = u4files.get_region_points(region, "Crumstadt", h5path)
    data = u4files.load_data_from_points(h5path, points)
    return data, points.crs


if __name__ == "__main__":
    main()
