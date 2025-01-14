"""Compares the detected ground motion with local GNSS stations"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from scipy import signal as spsig

import u4py.io.gpkg as u4gpkg
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4projects


def main():
    # Paths
    project = u4projects.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Examples\Compare_with_GPS_GPKG.u4project"
        ).expanduser(),
        required=["ext_path", "output_path"],
        interactive=False,
    )
    psi_path_bbd1 = os.path.join(
        project["paths"]["base_path"],
        "Data_2021",
        "INSAR_Data",
        "BBD_2021_vertikal.gpkg",
    )
    psi_path_bbd2 = os.path.join(
        project["paths"]["base_path"], "Data_2023", "hessen_l3_clipped.gpkg"
    )

    output_folder = os.path.join(
        project["paths"]["output_path"], "Compare_2021_2023"
    )
    os.makedirs(output_folder, exist_ok=True)

    data_1 = u4convert.reformat_gdf_to_dict(
        u4gpkg.load_gpkg_data_where_ogr(
            psi_path_bbd1, "BBD_2021_vertikal", "ID==1964510"
        )
    )
    data_2 = u4convert.reformat_gdf_to_dict(
        u4gpkg.load_gpkg_data_where_ogr(
            psi_path_bbd2, "vertikal", "ID==1442325"
        )
    )
    overlapping_data = data_2["timeseries"][0][
        data_2["time"] <= data_1["time"][-2]
    ]
    difference = data_1["timeseries"][0] - overlapping_data
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6 * 1.5, 3 * 1.5))
    axes[0].plot(
        data_1["time"], data_1["timeseries"][0], label="BBD 2015-2021"
    )
    axes[0].plot(
        data_2["time"], data_2["timeseries"][0], label="BBD 2015-2022"
    )
    axes[1].plot(
        data_1["time"],
        difference,
        color="k",
        label="Differenz",
    )
    avg_diff = unc.ufloat(np.nanmean(difference), 2 * np.nanstd(difference))
    axes[1].annotate(
        f"Durchschnitt: {avg_diff} mm".replace("+/-", "$\\pm$"),
        (0.975, 0.05),
        xycoords="axes fraction",
        horizontalalignment="right",
    )

    axes[0].set_ylabel("Bodenbewegung (mm)")
    axes[1].set_ylabel("Differenz (mm)")
    axes[1].set_xlabel("Zeitraum")
    fig.legend(loc="lower center", ncols=3)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.175, top=0.99)
    fig.savefig(os.path.join(output_folder, "BBD_21_vs_23.png"))
    fig.savefig(os.path.join(output_folder, "BBD_21_vs_23.pdf"))

    fig2, ax2 = plot_xcor(
        data_1["time"],
        data_1["timeseries"][0],
        data_2["time"],
        data_2["timeseries"][0],
    )
    fig2.savefig(os.path.join(output_folder, "BBD_xcorr.png"))
    fig2.savefig(os.path.join(output_folder, "BBD_xcorr.pdf"))


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
