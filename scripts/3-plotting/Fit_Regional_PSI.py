""" Lets the user define a region and fits all PSI points in it """

import os
import string
from datetime import datetime, timedelta

import contextily
import geopandas as gp
import matplotlib.gridspec as gs
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from shapely.geometry import Polygon

import u4py.analysis.other as u4other
import u4py.utils.files as u4files


def main():
    psivert_path = r"C:\Users\Michael Rudolf\Documents\ArcGIS\INSAR_Data\BBD_2021_PSI_Vertikal.h5"
    damage_shp = gp.GeoDataFrame.from_file(
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\Gebaudeschaeden.shp"
    )
    # Define osm query
    query = {
        "address": "Crumstadt",
        "tags": {
            "landuse": ["residential", "industrial"],
            "amenity": "hospital",
        },
    }
    # Get points from file, if not available creates new file
    points = u4files.get_select_points(query, psivert_path)
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
        crs=points.crs,
    )
    points_region = u4files.get_region_points(
        region, "Crumstadt", psivert_path
    )
    points_well = u4files.get_point_points(
        (464350, 5516100), 300, "OilWell", psivert_path
    )

    ind = points.source_ind.to_numpy()
    ind_region = points_region.source_ind.to_numpy()
    ind_well = points_well.source_ind.to_numpy()
    psi_data = u4files.load_hdf5(psivert_path, ind=ind)
    psi_data_region = u4files.load_hdf5(psivert_path, ind=ind_region)
    psi_data_well = u4files.load_hdf5(psivert_path, ind=ind_well)
    slope = get_linfit_each(psi_data_region)

    damage_shp = damage_shp.to_crs(points.crs)
    fig = plot_map_and_timeseries(
        psi_data, psi_data_region, slope, psi_data_well, points, damage_shp
    )
    fig_path = os.path.join(
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\selected_psi_points",
        query["address"],
    )
    # plt.show()
    fig.savefig(fig_path)
    fig.savefig(fig_path + ".pdf")


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


def plot_map_and_timeseries(
    psi_data, psi_data_region, slope, psi_data_well, points, damage_shape
):
    fig = plt.figure(figsize=(12, 6), dpi=150)
    grid = gs.GridSpec(ncols=2, nrows=2)
    axes = [
        fig.add_subplot(grid[:, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    ]
    # fig, axes = plt.subplots(ncols=2, figsize=(12, 6), dpi=150)
    numerate_axes(fig)
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
    damage_shape.plot(
        ax=axes[0],
        color="k",
        # markersize=,
        label="Known Building Damage",
        marker="x",
        markersize=8,
    )
    axes[0].set_xlim(463200, 469500)
    axes[0].set_ylim(5514800, 5520400)
    # axes[0].axis("equal")
    axes[0].set_xlabel("Longitude (m)")
    axes[0].set_ylabel("Latitude (m)")
    txt = axes[0].annotate(
        "Crumstadt",
        (465500, 5518000),
        # xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",
    )
    txt.set_path_effects(
        [
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ]
    )
    txt = axes[0].annotate(
        "Hahn",
        (468700, 5516000),
        # xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",
    )
    txt.set_path_effects(
        [
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ]
    )
    txt = axes[0].annotate(
        "Oil Wells",
        (464500, 5516100),
        # xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontweight="bold",
    )
    txt.set_path_effects(
        [
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ]
    )
    axes[0].plot(
        464350,
        5516100,
        color="k",
        marker="^",
        linewidth=0,
        markersize=10,
    )

    axes[0].ticklabel_format(style="plain")
    yticks = axes[0].get_yticks()
    new_ticks = [yt for yt in yticks if np.remainder(yt, 2000) == 0]
    axes[0].set_yticks(new_ticks)
    yticks = axes[0].get_xticks()
    new_ticks = [yt for yt in yticks if np.remainder(yt, 2000) == 0]
    axes[0].set_xticks(new_ticks)
    axes[0].set_yticklabels(
        axes[0].get_yticks().astype(int),
        rotation=90,
        verticalalignment="center",
    )
    axes[0].legend(loc="upper right")
    add_timeseries(psi_data, axes[1], "Crumstadt", shareax=axes[2])
    add_timeseries(
        psi_data_well, axes[2], "Oil Wells", legend=False, shareax=axes[1]
    )
    contextily.add_basemap(
        axes[0],
        crs=points.crs.to_string(),
        # zoom=15,
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    # plt.subplots_adjust(right=0.99, top=0.99, bottom=0.07)
    fig.tight_layout()
    return fig


def add_timeseries(psi_data, ax, title, legend=True, shareax=False):
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

    time_days = np.linspace(0, (time[-1] - time[0]).days, len(time))

    lin_popt, lin_pcov = spopt.curve_fit(
        u4other.poly1,
        time_days,
        y_med,
    )
    linear_component = u4other.poly1(time_days, *lin_popt)
    y_detrend = y_med - linear_component

    popt, pcov = spopt.curve_fit(
        u4other.sinefunc,
        time_days,
        y_detrend,
        p0=[2, 6 / 365.25, 0],
    )
    sinus_component = u4other.sinefunc(time_days, *popt)
    # y_residual = y_detrend - sinus_component

    ax.plot(time, y_med, ".", label="Median")
    ax.plot(time, linear_component + sinus_component, label="Fit")
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
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Vertical Displacement (mm)")
    if legend:
        ax.legend(loc="upper left")
    if shareax:
        ax.sharex(shareax)
        ax.sharey(shareax)


def numerate_axes(fig, n=0, step=1):
    """Adds numbering to all axes in a figure"""
    axes = fig.get_axes()

    for ii in range(0, len(axes), step):
        axes[ii].annotate(
            "(" + string.ascii_lowercase[int((ii + n) / step)] + ")",
            (-0.1, 1.05),
            xycoords="axes fraction",
            fontweight="bold",
            fontsize="xx-large",
            verticalalignment="center",
            horizontalalignment="center",
            # bbox=dict(fc="w", boxstyle="Circle"),
        )


if __name__ == "__main__":
    main()
