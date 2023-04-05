""" Lets the user define a region and fits all PSI points in it """


import os
from datetime import datetime

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import u4py.addons.climate as u4climate
import u4py.addons.gas_storage as u4gas
import u4py.addons.rivers as u4rivers
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.files as u4files


def main():
    # Paths
    psi_source = "L3_BBD_Vert_2023"
    psivert_path = (
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Data_2023\BBD_Vert"
    )
    # psivert_path = r"C:\Users\Michael Rudolf\Documents\ArcGIS\Data_2021\INSAR_Data\BBD_2021_PSI_Vertikal.h5"
    fig_path = os.path.join(
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\selected_psi_points",
        "Crumstadt" + "_" + psi_source,
    )

    # Load Data
    data, crs = get_data_within_osm_query(psivert_path)
    data_region, _ = get_data_within_region(psivert_path, crs=crs)
    data_well, _ = get_data_at_well(psivert_path)
    date_rhine, level_rhine = u4rivers.load_rhine_date(
        r"C:\Users\Michael Rudolf\HESSENBOX-DA\Umwelt_4_privat\scripts\Wasserstand des Rheins bei Düsseldorf monatlich ab 1996.csv"
    )
    date_gas, level_gas = u4gas.load_gas_data(
        r"C:\Users\Michael Rudolf\HESSENBOX-DA\Umwelt_4_privat\scripts\Inventory Turnover Data_23.txt"
    )
    climate_data = u4climate.load_climate_data(
        r"C:\Users\Michael Rudolf\HESSENBOX-DA\Umwelt_4_privat\scripts\klarchiv_01420_month_his\produkt_klima_monat_19350701_20211231_01420.txt"
    )

    # Create the figure
    fig, axes = prepare_figure()

    # The map
    plot_map(axes, data_region, crs=crs)

    # Time series for Crumstadt
    add_timeseries(
        data, axes[1], "Ground Motion in Crumstadt", shareax=axes[2]
    )

    # Time series for the gas storage
    add_timeseries(
        data_well,
        axes[2],
        "Ground Motion at Gas Storage",
        legend=False,
        shareax=axes[1],
    )

    # Rainfall
    axes[3].bar(
        climate_data["time"], climate_data["rain"], width=10, color="C0"
    )
    axes[3].set_ylabel("Avg. Rainfall (mm)\n(Frankfurt)")
    axes[3].set_ylim(0, 130)

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

    # Water Level
    axes[5].plot(date_rhine, level_rhine, "s-", color="C1")
    axes[5].set_ylabel("Avg. Water Level (cm)\n(Rhine in Düsseldorf)")
    axes[5].set_ylim(0, 700)

    # Gas data
    axes[6].plot(date_gas, (level_gas / np.max(level_gas)) * 100, color="C2")
    axes[6].set_ylabel("Fill level of gas storage (%)")
    axes[6].set_ylim(
        0,
    )

    # Formatting and saving
    axes[1].set_xlim(
        datetime(2015, 1, 1),
        # datetime(2021, 1, 1),
    )
    for ii in range(2, len(axes)):
        axes[ii].sharex(axes[1])

    axes[0].legend(loc="upper right")

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
    points = u4files.get_select_points_osm(query, h5path)
    data = load_data_from_points(h5path, points)
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
    data = load_data_from_points(h5path, points)
    return data, points.crs


def get_data_at_well(h5path):
    points = u4files.get_point_points(
        (464350, 5516100), 300, "OilWell", h5path
    )
    data = load_data_from_points(h5path, points)
    return data, points.crs


def load_data_from_points(h5path, points):
    # Loading for a single file:
    if h5path.endswith(".h5"):
        ind = points.source_ind.to_numpy()
        data = u4files.load_hdf5(h5path, ind=ind)

    # Loading for a folder of split files (better parallelization)
    else:
        data_filelist = u4files.points_to_filelist(points, h5path)
        data = u4files.load_hdf5_list(data_filelist)
    return data


def prepare_figure():
    fig = plt.figure(figsize=(12, 12), dpi=150)
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


def plot_map(axes, data_region, crs):
    # Map
    u4ax.plot_region_trend(data_region, ax=axes[0])
    u4plotfmt.add_map_label("Crumstadt", (465500, 5518000), ax=axes[0])
    u4plotfmt.add_map_label("Hahn", (468700, 5516000), ax=axes[0])
    u4plotfmt.add_map_label("Gas Storage", (464500, 5516100), ax=axes[0])

    axes[0].plot(
        464350,
        5516100,
        color="k",
        marker="^",
        linewidth=0,
        markersize=10,
    )

    u4ax.add_shapefile(
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\Gebaudeschaeden.shp",
        crs=crs,
        ax=axes[0],
        color="k",
        # markersize=,
        label="Known Building Damage",
        marker="x",
        markersize=8,
    )

    u4ax.add_shapefile(
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\Tiefenlinie_Top_Sand_7.shp",
        ax=axes[0],
        column="Z",
        facecolor="none",
        zorder=1,
    )
    u4ax.add_shapefile(
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\Gas_Störungen.shp",
        ax=axes[0],
        color="k",
        label="Faults",
        zorder=1,
    )

    axes[0].set_xlim(463200, 469500)
    axes[0].set_ylim(5514800, 5520400)
    axes[0].set_xlabel("Longitude (m)")
    axes[0].set_ylabel("Latitude (m)")
    u4plotfmt.map_style_ticks(axes[0], divisor=2000)


def add_timeseries(data, ax, title, legend=True, shareax=False):
    u4ax.plot_timeseries(data["time"], data["timeseries"], ax=ax, color="C0")
    u4ax.plot_timeseries_fit(data, ax=ax)
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
