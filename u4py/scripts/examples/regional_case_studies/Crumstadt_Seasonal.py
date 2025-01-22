"""
Plots the regional ground motion in Crumstadt, Hessen and some additional data
for it.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

# import contextily
# import geopandas as gp
# import matplotlib as mpl
# import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import u4py.addons.climate as u4climate
import u4py.addons.gas_storage as u4gas
import u4py.addons.groundwater as u4gw
import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.analysis.processing as u4proc

# import u4py.analysis.spatial as u4spatial
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4plotfmt
import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4proj

# from matplotlib.figure import Figure
# from matplotlib.lines import Line2D


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
    data_well, region_well = u4psi.get_point_data(
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
    data, region_osm = u4psi.get_osm_data(query, psi_path, overwrite=overwrite)

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
    fig, axes = plt.subplots(
        figsize=(12, 12), dpi=300, nrows=3, ncols=2, sharex=True
    )
    axes = axes.flat
    print("Ground Motion in Crumstadt")
    add_timeseries(
        data,
        axes[0],
        # "Vertical Ground Motion in Crumstadt",
        shareax=axes[0],
        inversion_path=os.path.join(
            project["paths"]["processing_path"], "Crumstadt.pkl"
        ),
        overwrite=overwrite,
        color="C1",
    )

    print("Ground Motion at Gas Storage")
    add_timeseries(
        data_well,
        axes[1],
        # "Vertical Ground Motion at Gas Storage",
        shareax=axes[0],
        inversion_path=os.path.join(
            project["paths"]["processing_path"], "GasStorage.pkl"
        ),
        overwrite=overwrite,
        color="C2",
    )

    # Rainfall
    axes[2].bar(
        climate_data["time"], climate_data["rain"], width=15, color="C0"
    )
    axes[2].set_ylabel("Avg. Rainfall (mm)\n(Frankfurt)")
    axes[2].set_ylim(0, 130)
    u4plotfmt.add_copyright("Source: DWD", axes[2])

    # Temperature data
    add_temp_timeseries(
        axes[3], temp_data, selected_stations=selected_stations
    )

    # Water Level
    add_waterlevel(axes[4], data_gw, station_list, label_list)

    # Gas data
    axes[5].plot(date_gas, level_gas, color="k", linewidth=2)
    axes[5].set_ylabel("Fill level of gas storage (%)")
    axes[5].set_ylim(
        0,
    )
    u4plotfmt.add_copyright("Source: MND Energies", ax=axes[5])

    # Formatting and saving
    axes[1].set_xlim(
        datetime(2015, 1, 1),
        datetime(2022, 2, 1),
    )
    axes[1].set_ylim(-10, 25)
    axes[-2].set_xlabel("Year")
    axes[-1].set_xlabel("Year")
    # plt.show()
    u4plotfmt.enumerate_axes(fig)
    fig.tight_layout()
    fig.savefig(fig_path)
    fig.savefig(fig_path + ".pdf")


def add_timeseries(
    data: dict,
    ax: Axes,
    title: str = "",
    legend: bool = True,
    shareax: Axes = None,
    inversion_path: os.PathLike = "",
    overwrite: bool = False,
    color="C0",
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
    results = u4proc.get_psi_dict_inversion(
        data, save_path=inversion_path, overwrite=overwrite
    )
    u4plotprep.print_inversion_results_for_publications(results)
    u4ax.plot_timeseries_fit(
        ax=ax, results=results, color=color, color_fit="k"
    )
    ax.set_title(title)
    ax.set_ylabel("Vertical Displacement (mm)")
    if legend:
        ax.legend(loc="upper center", ncols=3)
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
    y_stat = np.array([data[station] for station in selected_stations])
    y_stat[y_stat < -40] = np.nan
    y_mean = np.nanmean(y_stat, axis=0)
    ax.plot(
        data["time"],
        y_mean,
        ".",
        color="C3",
        markersize=3,
        label="Daily Mean",
    )

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
    ax.set_ylabel("Temperature (°C)")
    ax.legend(
        loc="upper center",
        ncols=3,
    )
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
    colors = [(b, b, 1) for b in np.linspace(0, 0.8, 4)]
    for ii, (station, label) in enumerate(zip(station_list, label_list)):
        ax.plot(
            data_gw[station]["time"],
            data_gw[station]["height"],
            ".",
            markersize=5,
            color=colors[ii],
            label=label,
        )
    ax.set_ylabel("Ground Water Level\n(m a.s.l)")
    # ax.set_ylim(0, 700)
    u4plotfmt.add_copyright("Source: HLNUG", ax)
    ax.set_ylim(84, 90)
    ax.legend(loc="upper center", ncols=2, columnspacing=1, handletextpad=0.2)


if __name__ == "__main__":
    main()
