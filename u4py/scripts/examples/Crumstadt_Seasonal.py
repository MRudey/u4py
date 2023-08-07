"""
Plots the regional ground motion in Crumstadt, Hessen and some additional data
for it.
"""


import os
from datetime import datetime
from typing import Tuple

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import Polygon

import u4py.addons.climate as u4climate
import u4py.addons.gas_storage as u4gas

# import u4py.addons.rivers as u4rivers
import u4py.addons.groundwater as u4gw
import u4py.analysis.processing as u4proc
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj


def main():
    # Paths
    psi_source = "L3_BBD_Vert_2023"
    project = u4proj.get_project(
        required=[
            "base_path",
            "psivert_path",
            "ext_path",
            "places_path",
            "output_path",
            "processing_path",
        ],
        interactive=False,
    )
    fig_path = os.path.join(
        project["paths"]["output_path"], "Crumstadt_" + psi_source
    )

    # Load Data
    data, crs = get_data_within_osm_query(project["paths"]["psivert_path"])
    data_region, _ = get_data_within_region(
        project["paths"]["psivert_path"], crs=crs
    )
    data_well, _ = get_data_at_well(project["paths"]["psivert_path"])
    # date_rhine, level_rhine = u4rivers.load_rhine_date(
    #     os.path.join(project["paths"]["ext_path"], "Wasserstand_Rhein_DD.csv")
    # )
    data_gw = u4gw.get_groundwater_data(
        os.path.join(
            project["paths"]["ext_path"], "GWStände_2015", "GWStände_2015.pkl"
        )
    )
    date_gas, _, level_gas = u4gas.load_gas_data(
        os.path.join(
            project["paths"]["ext_path"], "Inventory Turnover Data_23.txt"
        )
    )
    climate_data = u4climate.load_climate_data(
        os.path.join(project["paths"]["ext_path"], "Wetter_FFM.txt")
    )

    # Create the figure
    fig, axes = prepare_figure()

    # The map
    plot_map(
        axes[0],
        data_region,
        crs=crs,
        places_path=project["paths"]["places_path"],
    )

    # Time series for Crumstadt
    add_timeseries(
        data,
        axes[1],
        "Ground Motion in Crumstadt",
        shareax=axes[2],
        inversion_path=os.path.join(
            project["paths"]["processing_path"], "Crumstadt.pkl"
        ),
    )

    # Time series for the gas storage
    add_timeseries(
        data_well,
        axes[2],
        "Ground Motion at Gas Storage",
        legend=False,
        shareax=axes[1],
        inversion_path=os.path.join(
            project["paths"]["processing_path"], "GasStorage.pkl"
        ),
    )

    # Rainfall
    axes[3].bar(
        climate_data["time"], climate_data["rain"], width=10, color="C0"
    )
    axes[3].set_ylabel("Avg. Rainfall (mm)\n(Frankfurt)")
    axes[3].set_ylim(0, 130)
    u4plotfmt.add_copyright("Source: DWD", axes[3])

    # Temperature data
    axes[4].plot(climate_data["time"], climate_data["mean"], "s-", color="C3")
    axes[4].fill_between(
        climate_data["time"],
        climate_data["min"],
        climate_data["max"],
        color="C3",
        alpha=0.5,
        edgecolor=None,
    )
    axes[4].set_ylabel("Avg. Temperature (°C)\n(Frankfurt)")
    u4plotfmt.add_copyright("Source: DWD", axes[4])

    # Water Level
    axes[5].plot(
        data_gw["CRUMSTADT"]["time"],
        data_gw["CRUMSTADT"]["height"],
        ".-",
        color="C0",
        label="Crumstadt",
    )
    axes[5].plot(
        data_gw["HAHN flach"]["time"],
        data_gw["HAHN flach"]["height"],
        ".-",
        color="C1",
        label="Hahn (shallow)",
    )
    axes[5].plot(
        data_gw["ALLMENDFELD (alt)"]["time"],
        data_gw["ALLMENDFELD (alt)"]["height"],
        ".-",
        color="C2",
        label="Allmendfeld (old)",
    )
    axes[5].set_ylabel("Ground Water Level (m)")
    # axes[5].set_ylim(0, 700)
    u4plotfmt.add_copyright("Source: HLNUG", axes[5])
    axes[5].set_ylim(85, 90)
    axes[5].legend(loc="best")

    # Gas data
    axes[6].plot(date_gas, level_gas, color="C2")
    axes[6].set_ylabel("Fill level of gas storage (%)")
    axes[6].set_ylim(
        0,
    )
    u4plotfmt.add_copyright("Source: MND Energies", ax=axes[6])

    # Formatting and saving
    axes[1].set_xlim(
        datetime(2015, 1, 1),
        # datetime(2021, 1, 1),
    )
    for ii in range(2, len(axes)):
        axes[ii].sharex(axes[1])

    # plt.show()
    contextily.add_basemap(
        axes[0],
        crs=crs,
        # zoom=15,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    fig.tight_layout()
    fig.savefig(fig_path)
    fig.savefig(fig_path + ".pdf")


def get_data_within_osm_query(h5path: os.PathLike) -> Tuple[dict, str]:
    """Defines the OSM query for Crumstadt and loads the data from the region.

    :param h5path: The path to the h5file.
    :type h5path: os.PathLike
    :return: A tuple containing (data, crs).
    :rtype: Tuple[dict, str]
    """
    # Define osm query
    query = {
        "address": "Crumstadt",
        "tags": {
            "landuse": ["residential", "industrial"],
            "amenity": "hospital",
        },
    }
    # Get points from file, if not available creates new file
    points = u4files.get_select_points_osm(query, h5path)
    data = u4files.load_data_from_points(h5path, points)
    return data, points.crs


def get_data_within_region(h5path: os.PathLike, crs: str) -> Tuple[dict, str]:
    """Gets the data in the region of the map's extend for plotting as points on the map.

    :param h5path: The path to the data.
    :type h5path: os.PathLike
    :param crs: The target crs.
    :type crs: str
    :return:  A tuple containing (data, crs).
    :rtype: Tuple[dict, str]
    """
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


def get_data_at_well(h5path: os.PathLike) -> Tuple[dict, str]:
    """Gets the data in a radius of 300 m around the well.

    :param h5path: The path to the data.
    :type h5path: os.PathLike
    :return: A tuple containing (data, crs)
    :rtype: Tuple[dict, str]
    """
    points = u4files.get_point_points(
        (464350, 5516100), 500, "OilWell", h5path
    )
    data = u4files.load_data_from_points(h5path, points)
    return data, points.crs


def prepare_figure() -> Tuple[Figure, Axes]:
    """Generates a gridded plot with suplots that span several rows.

    :return: The figure and axis in a tuple.
    :rtype: Tuple[Figure, Axes]
    """
    fig = plt.figure(figsize=(12, 12), dpi=300)
    grid = gs.GridSpec(ncols=2, nrows=4)
    axes = [
        fig.add_subplot(grid[:2, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, 0]),
        fig.add_subplot(grid[3, 1]),
    ]
    u4plotfmt.numerate_axes(fig)
    return fig, axes


def plot_map(ax: Axes, data_region: dict, crs: str, places_path: os.PathLike):
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

    u4ax.add_shapefile(
        os.path.join(places_path, "GW_Stations.shp"),
        ax=ax,
        marker=u4plotfmt.drop_shape(),
        color="b",
        markersize=50,
        zorder=1,
        label="Groundwater Wells",
        keys=["CRUMSTADT", "HAHN flach", "ALLMENDFELD (alt)"],
        labels=["Crumstadt", "Hahn\n(shallow)", "Allmendfeld\n(old)"],
    )

    ax.set_xlim(463200, 469500)
    ax.set_ylim(5514000, 5520400)
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    u4plotfmt.map_style(ax, divisor=1000, crs=crs)
    ax.legend(loc="upper right")


def add_timeseries(
    data: dict,
    ax: Axes,
    title: str = "",
    legend: bool = True,
    shareax: Axes = None,
    inversion_path: os.PathLike = "",
    overwrite: bool = False,
):
    """Adds a timeseries with fit to the given axis.

    :param data: The data to add.
    :type data: dict
    :param ax: The axis to add the data to.
    :type ax: Axes
    :param title: The title of the plot, defaults to ""
    :type title: str
    :param legend: Adds a legend to the plot, defaults to True
    :type legend: bool, optional
    :param shareax: The axis to share the axis limits with, defaults to None
    :type shareax: Axes, optional
    :param inversion_path: The path to the inversion data file for this region, defaults to ""
    :type inversion_path: os.PathLike, optional
    """
    results = u4proc.invert_psi_dict(
        data, save_path=inversion_path, overwrite=overwrite
    )

    u4ax.plot_timeseries_fit(ax=ax, results=results)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Vertical Displacement (mm)")
    if legend:
        ax.legend(loc="upper left")
    if shareax:
        ax.sharex(shareax)
        ax.sharey(shareax)


if __name__ == "__main__":
    main()
