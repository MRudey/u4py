""" Lets the user define a region and fits all PSI points in it """

import locale
import os
import string
from datetime import datetime

import contextily
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt

import u4py.analysis.other as u4other
import u4py.utils.files as u4files


def main():
    psi_vert_path = u4files.get_file_paths(
        filetype=((".h5", ".h5"),), title="Select PSI Data file"
    )
    region_path = u4files.get_file_paths(
        filetype=((".shp", ".shp"),), title="Select region shape file."
    )

    # Get points from file, if not available creates new file
    region = gp.read_file(region_path)

    points_region, point_files_folder = u4files.get_region_points(
        region, "FFM", psi_vert_path
    )

    ind_region = points_region.source_ind.to_numpy()
    psi_data_region = u4files.load_hdf5(psi_vert_path, ind=ind_region)
    slope = get_linfit_each(psi_data_region)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5), dpi=150)
    plot_map(axes, psi_data_region, slope)

    add_timeseries(
        psi_data_region, axes[1], "Ground Motion in FFM", include_fit=False
    )

    axes[0].legend(loc="upper right")
    axes[0].set_xlim(474000, 476000)
    axes[0].set_ylim(5550000, 5551500)
    # plt.show()
    contextily.add_basemap(
        axes[0],
        crs=region.crs.to_string(),
        # zoom=15,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )

    axes[0].set_xticklabels(format_labels(axes[0].get_xticklabels()))
    axes[0].set_yticklabels(format_labels(axes[0].get_yticklabels()))
    fig_path = os.path.join(point_files_folder, "FFM")
    fig.tight_layout()
    fig.savefig(fig_path)
    fig.savefig(fig_path + ".pdf")


def format_labels(ticks):
    for ii, vtxt in enumerate(ticks):
        v = int(vtxt.get_text())
        vtick_n = int(np.floor(v / 10000))
        vtick_exp = int((v - (vtick_n * 10000)) / 100)
        ticks[ii].set_text(f"${vtick_n}" + "^{" + f"{vtick_exp:02}" + "}$")
    return ticks


def get_linfit_each(psi_data):
    slope = []
    time = psi_data["time"]
    for y in psi_data["timeseries"]:
        slc = np.nonzero(np.isfinite(y))
        time_slice = time[slc]
        y = y[slc]
        time_days = np.linspace(0, len(time_slice) * 6, len(time_slice))

        lin_popt, lin_pcov = spopt.curve_fit(
            u4other.poly1,
            time_days,
            y,
        )
        slope.append(lin_popt[0] * 365)
    return slope


def plot_map(axes, psi_data_region, slope):
    # Map
    rng = np.percentile(np.abs(slope), 95)
    sc = axes[0].scatter(
        psi_data_region["x"],
        psi_data_region["y"],
        c=slope,
        vmin=-rng,
        vmax=rng,
        cmap="RdYlBu",
        label="PSI locations",
        s=5,
    )
    plt.colorbar(
        sc,
        ax=axes[0],
        orientation="horizontal",
        extend="both",
        label="Mean Vertical Velocity (mm/a)",
    )
    axes[0].set_xlabel("Longitude (m)")
    axes[0].set_ylabel("Latitude (m)")
    axes[0].ticklabel_format(style="plain")


def add_timeseries(
    psi_data, ax, title, legend=True, shareax=False, include_fit=True
):
    # Timeseries
    time = psi_data["time"]
    y_med = np.nanmedian(psi_data["timeseries"], axis=0)
    slc = np.nonzero(np.isfinite(y_med))
    time = time[slc]
    y_med = y_med[slc]
    y_95 = np.nanpercentile(psi_data["timeseries"], q=95, axis=0)[slc]
    y_68 = np.nanpercentile(psi_data["timeseries"], q=68, axis=0)[slc]
    y_32 = np.nanpercentile(psi_data["timeseries"], q=32, axis=0)[slc]
    y_5 = np.nanpercentile(psi_data["timeseries"], q=5, axis=0)[slc]

    if include_fit:
        time_days = np.linspace(0, (time[-1] - time[0]).days, len(time))

        lin_popt, lin_pcov = spopt.curve_fit(
            u4other.poly1,
            time_days,
            y_med,
        )
        linear_component = u4other.poly1(time_days, *lin_popt)
        y_detrend = y_med - linear_component

        popt, pcov = spopt.curve_fit(
            u4other.cosinefunc,
            time_days,
            y_detrend,
            p0=[2, 6 / 365.25, 0],
        )
        sinus_component = u4other.cosinefunc(time_days, *popt)
        # y_residual = y_detrend - sinus_component

        ax.plot(time, linear_component + sinus_component, label="Fit")
        ax.annotate(
            "Longterm Trend: %.1f mm/a\nYearly Variation: $\\pm$%.1f mm"  # \nPeriod: %i days"  # \nMaximum: %s %s"
            % (
                lin_popt[0] * 365,
                np.abs(popt[0]),
                # (1 / popt[1]) * 6,
                # prefix,
                # shift.strftime("%B"),
            ),
            (0.99, 0.01),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="bottom",
        )

    ax.plot(time, y_med, ".", label="Median")
    ax.fill_between(
        time,
        y_95,
        y_5,
        color="C0",
        alpha=0.5,
        edgecolor=None,
        label="Data Range",
    )
    ax.fill_between(time, y_68, y_32, color="C0", alpha=0.5, edgecolor=None)
    # shift = time[0] - timedelta(days=popt[2])

    # if shift.day > 10:
    #     prefix = "Mid"
    # elif shift.day > 20:
    #     prefix = "End"
    # else:
    #     prefix = "Start"

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Vertical Displacement (mm)")
    if legend:
        ax.legend()
    if shareax:
        ax.sharex(shareax)
        ax.sharey(shareax)


if __name__ == "__main__":
    main()
