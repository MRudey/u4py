"""
Plots the regional ground motion in Crumstadt, Hessen and some additional data
for it.
"""

import os

# from datetime import datetime, timedelta
from pathlib import Path

# import contextily
# import geopandas as gp
# import matplotlib as mpl
# import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

import u4py.addons.climate as u4climate
import u4py.addons.gas_storage as u4gas
import u4py.addons.groundwater as u4gw
import u4py.analysis.inversion as u4invert
import u4py.analysis.processing as u4proc

# import u4py.analysis.spatial as u4spatial
import u4py.io.psi as u4psi

# import u4py.plotting.axes as u4ax
# import u4py.plotting.formatting as u4plotfmt
# import u4py.plotting.preparation as u4plotprep
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4proj

# from matplotlib.axes import Axes


# from typing import Tuple


# import u4py.analysis.inversion as u4invert
# import u4py.analysis.other as u4other
# import u4py.analysis.processing as u4proc


# from matplotlib.figure import Figure
# from matplotlib.lines import Line2D


def main():
    # Paths
    psi_source = "hessen_l3_clipped"
    crs = "EPSG:32632"
    overwrite = False
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
    # fig_path = os.path.join(
    #     project["paths"]["output_path"], f"Crumstadt_{psi_source}"
    # )

    # Load Data
    data_well, region_well = u4psi.get_point_data(
        (464350, 5516100),
        500,
        "OilWell",
        psi_path,
        overwrite=overwrite,
    )
    res_well = u4proc.get_psi_dict_inversion(
        data_well,
        save_path=os.path.join(
            project["paths"]["processing_path"], "GasStorage.pkl"
        ),
        overwrite=overwrite,
    )
    query = {
        "address": "Crumstadt",
        "tags": {
            "landuse": ["residential", "industrial"],
            "amenity": "hospital",
        },
    }
    data_crum, region_osm = u4psi.get_osm_data(
        query, psi_path, overwrite=overwrite
    )
    res_crum = u4proc.get_psi_dict_inversion(
        data_crum,
        save_path=os.path.join(
            project["paths"]["processing_path"], "Crumstadt.pkl"
        ),
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
    y_stat = np.array([temp_data[station] for station in selected_stations])
    y_stat[y_stat < -40] = np.nan
    y_mean = np.nanmean(y_stat, axis=0)
    temp_ref = u4invert.reformat_simple_timeseries(
        np.array(temp_data["time"]), y_mean
    )
    res_temp = u4proc.get_psi_dict_inversion(
        temp_ref,
        save_path=os.path.join(
            project["paths"]["processing_path"], "Temperature.pkl"
        ),
        overwrite=overwrite,
    )

    gas_ref = u4invert.reformat_simple_timeseries(date_gas, level_gas)
    res_gas = u4proc.get_psi_dict_inversion(
        gas_ref,
        save_path=os.path.join(
            project["paths"]["processing_path"], "Gas_Reform.pkl"
        ),
        overwrite=True,
    )

    station_list = [
        "CRUMSTADT",
        "HAHN flach",
        "ALLMENDFELD (alt)",
        "GODDELAU",
    ]

    fig, ax = plot_xcor(
        data_well["time"],
        # res_well["t_fit_2"],
        np.nanmedian(data_well["timeseries"], axis=0),
        # res_well["U"]["y_fit_2_err"],
        np.array(temp_data["time"]),
        # res_temp["t_fit_2"],
        y_mean,
        # res_temp["U"]["y_fit_2_err"],
        y1_label="Vertical Displacement (mm)",
        col1="C2",
        y2_label="Temperature (°C)",
        col2="C3",
    )
    fig.savefig(
        os.path.join(
            project["paths"]["output_path"], "Well_vs_Temperature.pdf"
        )
    )
    plt.close(fig)

    fig, ax = plot_xcor(
        data_crum["time"],
        # res_crum["t_fit_2"],
        np.nanmedian(data_crum["timeseries"], axis=0),
        # res_crum["U"]["y_fit_2_err"],
        np.array(temp_data["time"]),
        # res_temp["t_fit_2"],
        y_mean,
        # res_temp["U"]["y_fit_2_err"],
        y1_label="Vertical Displacement (mm)",
        col1="C1",
        y2_label="Temperature (°C)",
        col2="C3",
    )
    fig.savefig(
        os.path.join(
            project["paths"]["output_path"], "Crumstadt_vs_Temperature.pdf"
        )
    )
    plt.close(fig)

    fig, ax = plot_xcor(
        # data_well["time"],
        res_well["t_fit_2"],
        # np.nanmedian(data_well["timeseries"], axis=0),
        res_well["U"]["y_fit_2_err"],
        date_gas,
        # res_gas["t_fit_1"],
        level_gas,
        # res_gas["U"]["y_fit_1_err"],
        y1_label="Residual Displacement (mm)",
        col1="C2",
        y2_label="Fill Level (%)",
        col2="k",
    )
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "Well_vs_Storage.pdf")
    )
    plt.close(fig)

    fig, ax = plot_xcor(
        # data_crum["time"],
        res_crum["t_fit_2"],
        # np.nanmedian(data_crum["timeseries"], axis=0),
        res_crum["U"]["y_fit_2_err"],
        date_gas,
        # res_gas["t_fit_1"],
        level_gas,
        # res_gas["U"]["y_fit_1_err"],
        y1_label="Residual Displacement (mm)",
        col1="C1",
        y2_label="Fill Level (%)",
        col2="k",
    )
    fig.savefig(
        os.path.join(
            project["paths"]["output_path"], "Crumstadt_vs_Storage.pdf"
        )
    )
    plt.close(fig)

    fig, ax = plot_xcor(
        # data_well["time"],
        res_well["t_fit_2"],
        # np.nanmedian(data_well["timeseries"], axis=0),
        res_well["U"]["y_fit_2_err"],
        np.array(climate_data["time"]),
        np.array(climate_data["rain"]),
        y1_label="Vertical Displacement (mm)",
        col1="C2",
        y2_label="Precipitation (mm)",
        col2="C0",
    )
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "Well_vs_Rainfall.pdf")
    )
    plt.close(fig)

    fig, ax = plot_xcor(
        # data_crum["time"],
        res_crum["t_fit_2"],
        # np.nanmedian(data_crum["timeseries"], axis=0),
        res_crum["U"]["y_fit_2_err"],
        np.array(climate_data["time"]),
        np.array(climate_data["rain"]),
        y1_label="Vertical Displacement (mm)",
        col1="C2",
        y2_label="Precipitation (mm)",
        col2="C0",
    )
    fig.savefig(
        os.path.join(
            project["paths"]["output_path"], "Crumstadt_vs_Rainfall.pdf"
        )
    )
    plt.close(fig)

    for stn in station_list:
        fig, ax = plot_xcor(
            # data_well["time"],
            res_well["t_fit_2"],
            # np.nanmedian(data_well["timeseries"], axis=0),
            res_well["U"]["y_fit_2_err"],
            np.array(data_gw[stn]["time"]),
            np.array(data_gw[stn]["height"]),
            y1_label="Vertical Displacement (mm)",
            col1="C2",
            y2_label="Ground Water Height (m)",
            col2="b",
        )
        fig.savefig(
            os.path.join(project["paths"]["output_path"], f"Well_vs_{stn}.pdf")
        )
        plt.close(fig)

        fig, ax = plot_xcor(
            # data_crum["time"],
            res_crum["t_fit_2"],
            # np.nanmedian(data_crum["timeseries"], axis=0),
            res_crum["U"]["y_fit_2_err"],
            np.array(data_gw[stn]["time"]),
            np.array(data_gw[stn]["height"]),
            y1_label="Vertical Displacement (mm)",
            col1="C1",
            y2_label="Ground Water Height (m)",
            col2="b",
        )
        fig.savefig(
            os.path.join(
                project["paths"]["output_path"], f"Crumstadt_vs_{stn}.pdf"
            )
        )
        plt.close(fig)


def align_data(t1, y1, t2, y2):
    t1sort = np.argsort(t1)
    t1 = t1[t1sort]
    y1 = y1[t1sort]
    nany1 = np.isfinite(y1)
    t1 = t1[nany1]
    y1 = y1[nany1]
    t2sort = np.argsort(t2)
    t2 = t2[t2sort]
    y2 = y2[t2sort]
    nany2 = np.isfinite(y2)
    t2 = t2[nany2]
    y2 = y2[nany2]

    t1 = u4convert.get_floatyear(t1)
    t2 = u4convert.get_floatyear(t2)
    dt = np.max([np.mean(np.diff(t1)), np.mean(np.diff(t2))])
    tstart = np.max([t1[0], t2[0]])
    tend = np.min([t1[-1], t2[-1]])
    t = np.arange(tstart, tend, dt)
    sig1 = np.interp(t, t1, y1)
    sig2 = np.interp(t, t2, y2)
    return t, sig1, sig2


def plot_xcor(
    t1,
    y1,
    t2,
    y2,
    y1_label="Signal 1",
    y2_label="Signal 2",
    col1="C0",
    col2="C1",
    col3="C4",
):
    t, sig1, sig2 = align_data(t1, y1, t2, y2)
    xc12 = spsig.correlate(sig1, sig2)
    xc12 /= np.max(xc12)
    lags = (
        spsig.correlation_lags(len(sig1), len(sig2)) * np.diff(t)[0] * 365.25
    )
    max_lag = lags[np.argmax(xc12)]

    fig, axes = plt.subplots(nrows=3, figsize=(10, 7), dpi=300)
    axes[0].plot(t, sig1, "s-", color=col1)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel(y1_label)
    axes[1].plot(t, sig2, "s-", color=col2)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel(y2_label)
    axes[2].plot(lags, xc12, "s-", color=col3)
    axes[2].set_xlabel("Lag (days)")
    axes[2].axvline(max_lag, color="k")
    axes[2].annotate(
        f"{int(max_lag)} days",
        (max_lag, 0),
        (0.5, 0.5),
        textcoords="offset fontsize",
    )
    axes[2].set_ylabel("Cross Correlation ()")
    fig.tight_layout()

    return fig, axes


if __name__ == "__main__":
    main()
