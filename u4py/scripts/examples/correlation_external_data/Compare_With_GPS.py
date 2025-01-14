"""Compares the detected ground motion with local GNSS stations"""

import datetime
import os
from pathlib import Path

import geopandas as gp
import matplotlib.axes as mplax
import matplotlib.lines as mline
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as spimage
import scipy.stats as spstats
import uncertainties as unc

import u4py.addons.gnss as u4gnss
import u4py.io.files as u4files
import u4py.io.psi as u4psi
import u4py.plotting.axes as u4ax
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4projects


def main():
    # Options

    distance = 75  # Distance around gnss station to look for psis
    filter_width = 0  # median filter size, 0=nofilter
    take_diff = False  # Take differences between points to correlate
    overwrite = False
    id_selection = ["BADH00DEU", "KLOP00DEU", "FFMJ00DEU"]

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
    psi_path_egms1 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2015-2021",
        "EGMS_2015_2021_vertikal.gpkg",
    )
    psi_path_egms2 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2018-2022",
        "EGMS_2018_2022_vertikal.gpkg",
    )
    psi_path_list = [
        psi_path_bbd1,
        psi_path_bbd2,
        psi_path_egms1,
        psi_path_egms2,
    ]
    tables = [
        "BBD_2021_vertikal",
        "vertikal",
        "EGMS_2015_2021_vertikal",
        "EGMS_2018_2022_vertikal",
    ]
    output_folder = os.path.join(
        project["paths"]["output_path"], "Compare_With_GPS"
    )
    os.makedirs(output_folder, exist_ok=True)
    gnss_folder = os.path.join(project["paths"]["ext_path"], "GNSS_Data")
    gnss_file_list = u4files.get_file_list(
        filetype=".dat",
        folder_path=gnss_folder,
    )
    stations = u4gnss.get_station_coordinates(
        os.path.join(gnss_folder, "Stations")
    )
    sel = [stations[stations.ID == ii] for ii in id_selection]

    fig, axes = plt.subplots(
        nrows=len(sel), sharex=True, sharey=True, figsize=(12, 6)
    )
    for ii, station in enumerate(sel):
        gnss_data = u4convert.gnss_dat_to_dict(gnss_file_list[ii])
        gnss_t = gnss_data["gps_datetime"]
        gnss_v = gnss_data["res_up"]
        if filter_width:
            gnss_v = spimage.median_filter(gnss_v, filter_width)
        axes[ii].plot(gnss_t, gnss_v, "-", color="k")

        for cc, (psi_path, table) in enumerate(zip(psi_path_list, tables)):
            plot_psi_vs_gnss(
                gnss_data,
                psi_path,
                table,
                station,
                distance,
                axes[ii],
                output_folder,
                filter_width,
                take_diff,
                overwrite,
                color=f"C{cc}",
            )
    fig.legend(
        handles=[
            mline.Line2D([], [], color="k", label="GNSS"),
            mline.Line2D(
                [],
                [],
                marker=".",
                linestyle="None",
                color="C0",
                label="BBD 2015-2021",
            ),
            mline.Line2D(
                [],
                [],
                marker=".",
                linestyle="None",
                color="C1",
                label="BBD 2015-2022",
            ),
            mline.Line2D(
                [],
                [],
                marker=".",
                linestyle="None",
                color="C2",
                label="EGMS 2016-2021",
            ),
            mline.Line2D(
                [],
                [],
                marker=".",
                linestyle="None",
                color="C3",
                label="EGMS 2018-2023",
            ),
        ],
        fontsize="small",
        loc="lower center",
        ncols=5,
    )
    axes[1].set_ylabel("Bodenbewegung (mm)")
    axes[-1].set_xlabel("Zeitraum")
    axes[0].set_ylim(-15, 15)
    axes[0].set_xlim(
        datetime.datetime(2015, 1, 1),
        datetime.datetime(2022, 6, 1),
    )
    fig.subplots_adjust(top=0.95, left=0.075, right=0.99, bottom=0.125)

    fig.savefig(os.path.join(output_folder, "GNSS_PSI_timeseries"))
    fig.savefig(os.path.join(output_folder, "GNSS_PSI_timeseries.pdf"))
    # plot_gnss_and_psi(gnss_file_list)


def plot_psi_vs_gnss(
    gnss_data: dict,
    psi_path: os.PathLike,
    table: str,
    station: gp.GeoDataFrame,
    distance: float,
    ax: mplax.Axes,
    output_folder: os.PathLike,
    filter_width: int,
    take_diff: bool,
    overwrite: bool,
    color: str = "C0",
):
    psi_data, _ = u4psi.get_point_data(
        station.geometry,
        distance,
        f"{station.ID.values[0]}_GNSS",
        psi_path,
        table=table,
        overwrite=overwrite,
    )

    if psi_data:
        psi_t = psi_data["time"]
        psi_v = np.nanmedian(psi_data["timeseries"], axis=0)
        gnss_t = gnss_data["gps_datetime"]
        gnss_v = gnss_data["res_up"]
        psi_val = unc.ufloat(np.nanmedian(psi_v), 2 * np.nanstd(psi_v))
        gnss_val = unc.ufloat(np.nanmedian(gnss_v), 2 * np.nanstd(gnss_v))
        print(
            station.name.values[0],
            f"psi: {psi_val},  gnss: {gnss_val}",
        )
        if filter_width:
            gnss_v = spimage.median_filter(gnss_v, filter_width)

        correlate_time_series(
            psi_t,
            psi_v,
            gnss_t,
            gnss_v,
            station.name.values[0],
            output_folder,
        )

        u4ax.plot_timeseries(
            psi_data["time"], psi_data["timeseries"], ax=ax, color=color
        )

        ax.set_title(station.name.values[0])
    return


def subsample_gnss(data_vert, gnss_data):
    """Gets only the gnss data closest to psi obervations"""
    pti = []
    if isinstance(data_vert, dict):
        for psi_t in data_vert["time"]:
            pti.append(np.argmin(np.abs(gnss_data["gps_datetime"] - psi_t)))
    else:
        for psi_t in data_vert:
            pti.append(np.argmin(np.abs(gnss_data - psi_t)))
    pti = np.unique(pti)
    return pti


def resample_time_series(psi_t, psi_v, gnss_t, gnss_v):
    """
    Resamples both time series to start and stop at the same time and with
    equal intervals given by the one with lowest sampling frequency
    """

    min_t = np.max([psi_t[0], gnss_t[0]])
    max_t = np.min([psi_t[-1], gnss_t[-1]])

    slc = np.argwhere(psi_t >= min_t).flat
    psi_t = psi_t[slc]
    psi_v = psi_v[slc]
    slc = np.argwhere(psi_t <= max_t).flat
    psi_t = psi_t[slc]
    psi_v = psi_v[slc]

    slc = np.argwhere(gnss_t >= min_t).flat
    gnss_t = gnss_t[slc]
    gnss_v = gnss_v[slc]
    slc = np.argwhere(gnss_t <= max_t).flat
    gnss_t = gnss_t[slc]
    gnss_v = gnss_v[slc]

    pti = subsample_gnss(psi_t, gnss_t)
    gnss_t = gnss_t[pti]
    gnss_v = gnss_v[pti]

    gnss_v, psi_t, psi_v, gnss_t = remove_nonfinite(
        gnss_v, psi_t, psi_v, gnss_t
    )
    psi_v, gnss_t, gnss_v, psi_t = remove_nonfinite(
        psi_v, gnss_t, gnss_v, psi_t
    )
    return (gnss_t, gnss_v, psi_t, psi_v)


def remove_nonfinite(ref, x1, x2, x3):
    x1 = x1[np.isfinite(ref)]
    x2 = x2[np.isfinite(ref)]
    x3 = x3[np.isfinite(ref)]
    ref = ref[np.isfinite(ref)]
    return (ref, x1, x2, x3)


def correlate_time_series(psi_t, psi_v, gnss_t, gnss_v, name, output_path):
    """Does a simple correlation and returns correlation and lag"""
    gnss_t, gnss_v, psi_t, psi_v = resample_time_series(
        psi_t, psi_v, gnss_t, gnss_v
    )
    xy_cor = spstats.pearsonr(gnss_v, psi_v)
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    scatter_hist(gnss_v, psi_v, ax, ax_histx, ax_histy, "gnss", "psi")
    p = xy_cor.pvalue

    if p < 0.05:
        if p < 0.0001:
            pstr = "p<<0.05"
        else:
            pstr = "p<0.05"
    else:
        pstr = f"p={p:.2}"

    ax.annotate(
        "Pearson: %.2f (%s)" % (xy_cor.statistic, pstr),
        (0.99, 0.01),
        xycoords="axes fraction",
        horizontalalignment="right",
    )
    fig.suptitle(name)
    name = name.replace(" ", "")
    name = name.replace("/", "_")

    fig.set_constrained_layout(True)
    fig.savefig(os.path.join(output_path, f"cross_correlation_{name}"))
    fig.savefig(os.path.join(output_path, f"cross_correlation_{name}.pdf"))


def scatter_hist(x, y, ax, ax_histx, ax_histy, xlabel, ylabel):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, density=True)
    ax_histx.annotate(
        xlabel, (0.5, 1), xycoords="axes fraction", verticalalignment="bottom"
    )
    ax_histy.hist(y, bins=bins, orientation="horizontal", density=True)
    xlims = ax_histx.get_xlim()
    xq = np.linspace(xlims[0], xlims[1], 100)
    dist_x = spstats.norm.fit(x)
    dist_y = spstats.norm.fit(y)
    ax_histx.plot(xq, spstats.norm(*dist_x).pdf(xq))
    ax_histy.plot(spstats.norm(*dist_y).pdf(xq), xq)
    ax_histy.annotate(
        ylabel,
        (1, 0.5),
        xycoords="axes fraction",
        rotation=360 - 90,
        horizontalalignment="left",
    )


if __name__ == "__main__":
    main()
