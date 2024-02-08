"""
Plots the regional ground motion in Crumstadt, Hessen and some additional data
for it.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import Polygon

import u4py.addons.climate as u4climate
import u4py.addons.gas_storage as u4gas
import u4py.addons.groundwater as u4gw
import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.analysis.processing as u4proc
import u4py.io.files as u4files
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4proj


def main():
    # Paths
    psi_source = "hessen_l3_clipped"
    crs = "EPSG:32632"
    overwrite = True
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Examples\Crumstadt_Seasonal_GPKG.u4project"
        ).expanduser(),
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
    psi_path = os.path.join(
        project["paths"]["psivert_path"], f"{psi_source}.gpkg"
    )
    fig_path = os.path.join(
        project["paths"]["output_path"], f"Crumstadt_{psi_source}"
    )

    # Load Data
    data_well = u4psi.get_point_data(
        (464350, 5516100),
        500,
        "OilWell",
        psi_path,
        overwrite=overwrite,
    )
    query = {
        "address": "Crumstadt",
        "tags": {
            "landuse": ["residential", "industrial"],
            "amenity": "hospital",
        },
    }
    data = u4psi.get_osm_data(query, psi_path, overwrite=overwrite)
    region = gp.GeoDataFrame(
        {
            "geometry": [
                Polygon(
                    [
                        (462000, 5514000),
                        (462000, 5521000),
                        (470000, 5521000),
                        (470000, 5514000),
                    ]
                )
            ]
        },
        crs=crs,
    )
    data_region = u4psi.get_region_data(
        region,
        "Crumstadt_FullArea",
        psi_path,
        crs=crs,
        overwrite=overwrite,
    )

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

    temp_data = u4climate.get_temperature_data(
        os.path.join(
            project["paths"]["ext_path"],
            "Lufttemperatur",
            "TU_temp_TMW2015_2022.csv",
        )
    )

    selected_stations = ["Riedstadt", "Darmstadt", "Mörfelden", "Raunheim"]
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

    # Create the figure
    fig, axes = prepare_figure()

    # The map
    plot_map(
        axes[0],
        data_region,
        crs=crs,
        places_path=project["paths"]["places_path"],
        station_list=station_list,
        label_list=label_list,
    )

    print("Ground Motion in Crumstadt")
    add_timeseries(
        data,
        axes[1],
        "Ground Motion in Crumstadt",
        shareax=axes[2],
        inversion_path=os.path.join(
            project["paths"]["processing_path"], "Crumstadt.pkl"
        ),
        overwrite=overwrite,
    )

    print("Ground Motion at Gas Storage")
    add_timeseries(
        data_well,
        axes[2],
        "Ground Motion at Gas Storage",
        legend=False,
        shareax=axes[1],
        inversion_path=os.path.join(
            project["paths"]["processing_path"], "GasStorage.pkl"
        ),
        overwrite=overwrite,
    )

    # Rainfall
    axes[3].bar(
        climate_data["time"], climate_data["rain"], width=10, color="C0"
    )
    axes[3].set_ylabel("Avg. Rainfall (mm)\n(Frankfurt)")
    axes[3].set_ylim(0, 130)
    u4plotfmt.add_copyright("Source: DWD", axes[3])

    # Temperature data
    add_temp_timeseries(
        axes[4], temp_data, selected_stations=selected_stations
    )

    # Water Level
    add_waterlevel(axes[5], data_gw, station_list, label_list)

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
    u4plotfmt.enumerate_axes(fig)
    return fig, axes


def plot_map(
    ax: Axes,
    data_region: dict,
    crs: str,
    places_path: os.PathLike,
    station_list,
    label_list,
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

    # Map
    u4ax.plot_region_trend(data_region, ax=ax, fit=False, use_gdf=True)
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
        keys=station_list,
        labels=label_list,
    )

    ax.set_xlim(462000, 470000)
    ax.set_ylim(5514000, 5520400)
    u4plotfmt.map_style(ax, divisor=2000, crs=crs)
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
    u4plotprep.print_inversion_results_for_publications(results)
    u4ax.plot_timeseries_fit(ax=ax, results=results)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Vertical Displacement (mm)")
    if legend:
        ax.legend(loc="upper right")
    if shareax:
        ax.sharex(shareax)
        ax.sharey(shareax)


def add_temp_timeseries(ax: Axes, data: dict, selected_stations: list):
    """Adds the temperature time series to the axis.

    :param ax: The axis to add the plot.
    :type ax: Axes
    :param data: The data read from the climate addon.
    :type data: dict
    :param selected_stations: A list of stations for plotting.
    :type selected_stations: list
    """
    data_inv = dict()
    for station in selected_stations:
        y = np.array(data[station])
        t = np.array(data["time"])[y > -40]
        y = y[y > -40]

        data_inv[station] = u4invert.reformat_simple_timeseries(
            t, y, station=station
        )
        ax.plot(t, y, ".", label=station, markersize=3)
    stacked_data = u4invert.stack_data(data_inv)
    # stacked_data = u4invert.smooth_stacked_data(stacked_data)
    (
        matrix,
        data_out,
        time_vector,
        parameter_list,
    ) = u4invert.invert_time_series(stacked_data)
    t_fit = u4convert.get_datetime(data_out["t"])
    ax.plot(t_fit, data_out["ori_dhat_data"]["dhatU"], "k", label="Inversion")
    print("Temperature Data")
    ann_temp, peak_temp = u4other.superpose(matrix[2], -matrix[3])
    peak_time = (peak_temp / (2 * np.pi)) * 365
    peak_date = datetime(2015, 1, 1, 0, 0, 0) + timedelta(days=peak_time)
    print(f"Amplitude: {ann_temp:.1f}")
    print("Peak date", peak_date)
    ax.set_ylabel("Avg. Temperature (°C)")
    ax.legend(loc="upper right", ncols=3)
    ax.set_ylim(-15, 50)
    u4plotfmt.add_copyright("Source: DWD/HLNUG", ax)


def add_waterlevel(
    ax: Axes, data_gw: dict, station_list: list, label_list: list
):
    """Adds the water level data of the given stations to the axis.

    :param ax: The axis where to add the plots.
    :type ax: Axes
    :param data_gw: The groundwater level data as read by the groundwater addon.
    :type data_gw: dict
    :param station_list: A list of stations to use.
    :type station_list: list
    :param label_list: Labels for the list of the stations.
    :type label_list: list
    """
    for ii, (station, label) in enumerate(zip(station_list, label_list)):
        ax.plot(
            data_gw[station]["time"],
            data_gw[station]["height"],
            ".",
            markersize=3,
            color=f"C{ii}",
            label=label,
        )
    ax.set_ylabel("Ground Water Level (m a.s.l)")
    # ax.set_ylim(0, 700)
    u4plotfmt.add_copyright("Source: HLNUG", ax)
    ax.set_ylim(84, 90)
    ax.legend(loc="upper right", ncols=2)


if __name__ == "__main__":
    main()
