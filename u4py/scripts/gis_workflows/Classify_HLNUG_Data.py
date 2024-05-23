"""
Uses the internal shape file of HLNUG to automatically create maps and reports
of the known mass movements database of HLNUG
"""

import os
from multiprocessing import Pool
from pathlib import Path

import geopandas as gp
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4pltfmt
import u4py.plotting.plots as u4plots
import u4py.utils.config as u4config


def main():
    shp_path = Path(
        "~/Documents/umwelt4/Places/HLNUG Daten/SHP/RD_Rutschungen_gesamt.shp"
    ).expanduser()
    output_path = Path(
        "~/Documents/umwelt4/INSAR_plots/HLNUG_Plots"
    ).expanduser()
    os.makedirs(output_path, exist_ok=True)
    data = gp.read_file(shp_path)
    args = [(data, ii, output_path) for ii in data.AMT_NR_.to_list()]
    for arg in tqdm(args):
        topo_plot(*arg)
    # with Pool(u4config.cpu_count) as p:
    #     list(
    #         tqdm(
    #             p.imap_unordered(plot_wrapper, args),
    #             total=len(args),
    #             desc="Generating HLNUG Plots",
    #             leave=False,
    #         )
    #     )


def plot_wrapper(args: tuple):
    try:
        topo_plot(*args)
    except TypeError:
        print(f"Error for ID: {args[1]}")


def topo_plot(data: gp.GeoDataFrame, num: int, output_path: os.PathLike):
    polygon = data[data.AMT_NR_ == num]
    cx = polygon.centroid.x.values[0]
    cy = polygon.centroid.y.values[0]

    fig, ax = plt.subplots(figsize=(16 / 2.54, 11.3 / 2.54), dpi=300)
    polygon.plot(
        ax=ax, facecolor="None", edgecolor="C0", linewidth=2, zorder=3
    )
    # buffer_size = 3 * np.sqrt(polygon.area)
    buffer_size = 750
    polygon.buffer(buffer_size).plot(ax=ax, facecolor="None", edgecolor="None")
    ax.annotate(
        f"{num}",
        (cx, cy),
        horizontalalignment="center",
        verticalalignment="center",
        # fontsize="xx-large",
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2, foreground="C0")],
        color="w",
    )

    ax.set_position([0, 0, 1, 1])
    plt.axis("equal")
    ax.legend(
        handles=[
            mpatches.Patch(
                facecolor="None",
                edgecolor="C0",
                linewidth=2,
                label="Rutschung aus DGM",
            ),
            mpatches.Patch(
                # alpha=0.33,
                edgecolor="k",
                facecolor="None",
                linestyle=":",
                # linewidth=0.5,
                label="weitere Rutschungen",
            ),
            mlines.Line2D(
                [],
                [],
                linewidth=2,
                color="orange",
                path_effects=[
                    pe.Stroke(linewidth=3, foreground="sienna"),
                    pe.Normal(),
                ],
                label="Landesstraße",
            ),
        ],
    )
    plt.draw()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data[data.AMT_NR_ != num].plot(
        ax=ax,
        # alpha=0.33,
        edgecolor="k",
        facecolor="None",
        linestyle=":",
        # linewidth=0.5,
        zorder=2,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    u4pltfmt.add_scalebar(ax, width=0)
    u4ax.add_basemap(ax=ax)

    # Use HVBG WMS -> currently not working
    # u4ax.add_web_map_service(
    #     ax=ax,
    #     crs=str(polygon.crs),
    #     # wms_url="https://ows.terrestris.de/osm/service",
    #     # layer="OSM-WMS",
    # )
    fig.savefig(os.path.join(output_path, f"topo_{num:04}.png"))
    # fig.savefig(os.path.join(output_path, f"topo_{num}.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
