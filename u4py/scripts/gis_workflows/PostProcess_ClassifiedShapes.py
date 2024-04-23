"""
Uses the shapes classified by the main workflow and generates a report for
each anomaly.
"""

import configparser
import os
import pickle as pkl
import subprocess
import textwrap
import warnings
from multiprocessing import Pool
from typing import Tuple

import contextily
import geopandas as gp
import matplotlib.axes as mplax
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import u4py.addons.web_services as u4web
import u4py.analysis.spatial as u4spatial
import u4py.io.files as u4files
import u4py.io.gpkg as u4gpkg
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4pltfmt
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

warnings.filterwarnings("ignore")

GENERATE_PLOTS = False


def main():
    project = u4proj.get_project(
        proj_path="/home/rudolf/Documents/umwelt4/PostProcess_ClassifiedShapes.u4project",
        required=[
            "base_path",
            "places_path",
            "output_path",
            "results_path",
            "diff_plan_path",
        ],
        interactive=False,
    )
    use_parallel = True
    generate_pdf = True

    output_path = os.path.join(
        project["paths"]["output_path"], "Detailed_Maps"
    )
    os.makedirs(output_path, exist_ok=True)
    class_shp_fp = os.path.join(
        project["paths"]["results_path"], "Classified_Shapes.gpkg"
    )
    cls_shp_fp_filtered = os.path.join(
        project["paths"]["results_path"], "Filtered_Classified_Shapes.gpkg"
    )
    hlnug_path = os.path.join(
        project["paths"]["places_path"],
        "Classifier_shapes",
        "classifier_shapes.gpkg",
    )
    dem_path = os.path.join(
        os.path.split(project["paths"]["diff_plan_path"])[0],
        "DGM1_2021",
        "raster_2021",
    )
    contour_path = os.path.join(
        project["paths"]["diff_plan_path"],
        "thresholded_contours_all_shapes.gpkg",
    )

    # Read Data
    if not os.path.exists(cls_shp_fp_filtered):
        class_shp_gdf = gp.read_file(class_shp_fp)

        lands_f = class_shp_gdf.landslides_num_inside > 0
        karst_f = class_shp_gdf.karst_num_inside > 0
        rockf_f = class_shp_gdf.rockfall_num_inside > 0
        motor_f = class_shp_gdf.roads_has_motorway == "1"

        filter_all = np.logical_or(np.logical_or(lands_f, karst_f), rockf_f)

        filter_all_w_mw = np.logical_and(
            np.logical_or(filter_all, motor_f), np.logical_not(filter_all)
        )

        gdf_filtered = class_shp_gdf[filter_all]

        cached_locations = os.path.join(
            project["paths"]["results_path"], "cached_locations.txt"
        )
        if not os.path.exists(cached_locations):
            locations = u4spatial.reverse_geolocate(gdf_filtered)
            with open(cached_locations, "wt", encoding="utf-8") as cache:
                for loc in locations:
                    cache.write(loc + "\n")
        else:
            with open(
                cached_locations, "rt", encoding="utf-8", newline="\n"
            ) as cache:
                locations = cache.readlines()
        if len(locations) != len(gdf_filtered):
            locations = u4spatial.reverse_geolocate(gdf_filtered)
            with open(cached_locations, "wt", encoding="utf-8") as cache:
                for loc in locations:
                    cache.write(loc + "\n")

        gdf_filtered = gdf_filtered.assign(locations=locations)
        gdf_filtered.to_file(cls_shp_fp_filtered)
    else:
        gdf_filtered = gp.read_file(cls_shp_fp_filtered)

    # gdf_mw = class_shp_gdf[filter_all_w_mw]

    args = [
        (
            row,
            gdf_filtered.crs,
            output_path,
            hlnug_path,
            project,
            dem_path,
            contour_path,
        )
        for row in gdf_filtered.iterrows()
    ]

    if use_parallel:
        with Pool(u4config.cpu_count - 2) as p:
            list(
                tqdm(
                    p.imap_unordered(wrap_map_worker, args),
                    total=len(args),
                    desc="Generating Plots",
                    leave=False,
                )
            )

    else:
        for arg in tqdm(
            args,
            total=len(args),
            desc="Generating Plots",
            leave=False,
        ):
            wrap_map_worker(arg)

    if generate_pdf:
        tex_folder = os.path.join(output_path, "tex_includes")
        include_list = [
            os.path.join(tex_folder, fp)
            for fp in os.listdir(tex_folder)
            if fp.endswith("tex")
        ]
        tex = (
            "\\documentclass[\n"
            + "  ngerman,\n"
            + "  logofile=/home/rudolf/Documents/umwelt4/SelectedSites/Detailed_Maps/tuda_logo.pdf,\n"
            + "  accentcolor=8c,\n"
            + "]{tudapub}\n"
            + "\\usepackage{graphicx}\n"
            + "\\usepackage{subcaption}\n"
            + "\\usepackage{enumitem}\n"
            + "\\usepackage{wrapfig}\n"
            + "\\usepackage{float}\n"
            + "\\usepackage[ngerman]{babel}\n"
            # + "\\usepackage[margin=1in]{geometry}\n"
            + "\\begin{document}\n"
            + "\\title{Detektierte Anomalien}\n"
            + "\\subtitle{Atlas anomaler Bodenbewegungen in Hessen}\n"
            + "\\author{automatisch generierter Report aus U4Py}\n"
            + "\\date{\\today}\n"
            + "\\addTitleBox{Institut für Angewandte Geowissenschaften}\n\n"
            + "\\maketitle\n\n"
        )
        include_list.sort()
        for incl in include_list:
            tex += f"\\input{{{incl}}}\n\\clearpage\n"
        tex += "\\end{document}"

        report_path = os.path.join(output_path, "Site_Report.tex")
        with open(
            report_path, "wt", encoding="utf-8", newline="\n"
        ) as tex_file:
            tex_file.write(tex)

        subprocess.run(["pdflatex", f"{report_path}"])
        subprocess.run(["pdflatex", f"{report_path}"])
        subprocess.run(["pdflatex", f"{report_path}"])


def wrap_map_worker(args: tuple):
    """Parallel wrapper for `map_worker()`

    :param args: The function arguments.
    :type args: tuple
    """
    map_worker(*args)


def map_worker(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    hlnug_path: os.PathLike,
    project: configparser.ConfigParser,
    dem_path: os.PathLike,
    contour_path: os.PathLike,
):
    """Calls the various plotting and reporting functions.

    :param row: The index and entry of a row in a GeoDataFrame of classified shapes.
    :type row: tuple
    :param crs: The crs of the dataset.
    :type crs: str
    :param output_path: The path where the output shall be saved.
    :type output_path: os.PathLike
    :param hlnug_path: The path where HLNUG data is found.
    :type hlnug_path: os.PathLike
    :param project: The project's config.
    :type project: configparser.ConfigParser
    :param dem_path: The path to the digital elevation model.
    :type dem_path: os.PathLike
    :param contour_path: The path to the contour file.
    :type contour_path: os.PathLike
    """

    # pass
    make_tex_report(row, output_path, "tex_includes")
    xlim, ylim = make_detailed_map(
        row,
        crs,
        output_path,
        "known_features",
        hlnug_path,
        contour_path,
    )
    if GENERATE_PLOTS:
        make_sat_map(row, crs, output_path, "known_features", xlim, ylim)
        make_diffplan_map(
            row,
            crs,
            output_path,
            "known_features",
            project["paths"]["diff_plan_path"],
            xlim,
            ylim,
        )
        make_dem_map(
            row,
            crs,
            output_path,
            "known_features",
            dem_path,
            xlim,
            ylim,
        )
        make_slope_map(
            row,
            crs,
            output_path,
            "known_features",
            dem_path,
            xlim,
            ylim,
        )
        make_geology_map(
            row,
            crs,
            output_path,
            "known_features",
            xlim,
            ylim,
            os.path.join(project["paths"]["places_path"], "legend_GK25.pkl"),
        )
        make_hydrogeology_map(
            row,
            crs,
            output_path,
            "known_features",
            xlim,
            ylim,
            os.path.join(
                project["paths"]["places_path"], "legend_HUEK200.pkl"
            ),
        )
        make_topsoil_map(
            row,
            crs,
            output_path,
            "known_features",
            xlim,
            ylim,
            os.path.join(project["paths"]["places_path"], "legend_BFD50.pkl"),
        )


def make_geology_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    xlim: tuple,
    ylim: tuple,
    legend_path: os.PathLike,
):
    """Creates a map of the geological features around the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    :param legend_path: The path where the legend is found.
    :type legend_path: os.PathLike
    """
    output_path = os.path.join(output_path, suffix)
    shp_path = os.path.join(output_path, "HLNUG_queries")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(shp_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig = plt.figure(figsize=(10, 7), dpi=150)
    gs = fig.add_gridspec(ncols=2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1])
    axl.axis("off")

    shp_gdf.plot(ax=ax, fc="None", ec="C0", zorder=5)
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")

    with open(legend_path, "rb") as leg_file:
        uuid, label, _, facecolor, _, _ = pkl.load(leg_file)

    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    geology_data = u4web.query_hlnug(
        "geologie/gk25/MapServer",
        "Geologie (Kartiereinheiten)",
        region=region,
        suffix=f"_{row[1].group:05}",
        out_folder=shp_path,
    )
    if len(geology_data) > 0:
        fc = [
            (facecolor[uuid.index(tkeh)] if tkeh in uuid else (0.5, 0.5, 0.5))
            for tkeh in geology_data["TKEH"].to_list()
        ]
        untkeh = np.unique(geology_data["TKEH"].to_numpy())
        untkeh.sort()
        leg_handles = []
        leg_handles.append(
            mlines.Line2D([], [], color="C0", label="Bereich der Anomalie")
        )
        for tkeh in untkeh:
            try:
                ii = uuid.index(tkeh)
                facec = facecolor[ii]
            except ValueError:
                facec = (0.5, 0.5, 0.5)
            leg_handles.append(
                mpatches.Patch(fc=facec, label=textwrap.fill(label[ii], 50))
            )
        geology_data.plot(ax=ax, fc=fc, ec="k", linewidth=0.25, alpha=0.5)
        fault_data = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Tektonik (Liniendaten)",
            region=region,
            suffix=f"_{row[1].group:05}",
            out_folder=shp_path,
        )
        leg_handles.append(
            mlines.Line2D([], [], color="k", label="Störungen, inkl. vermutet")
        )
        fault_data.plot(ax=ax, color="k")
        axl.legend(
            handles=leg_handles,
            fontsize="small",
            markerscale=0.75,
        )
        u4ax.add_basemap(
            ax=ax,
            crs=geology_data.crs,
            source=contextily.providers.CartoDB.Voyager,
        )
        #     bbox_to_anchor=(1.5, 1),
        #     loc="upper right",
        #     borderaxespad=0.0,
        # )

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"{row[1].group:05}_GK25.png"))
        fig.savefig(os.path.join(output_path, f"{row[1].group:05}_GK25.pdf"))
    plt.close(fig)


def make_hydrogeology_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    xlim: tuple,
    ylim: tuple,
    legend_path: os.PathLike,
):
    """Creates a map of hydrogeological units in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    :param legend_path: The path where the legend is found.
    :type legend_path: os.PathLike
    """
    output_path = os.path.join(output_path, suffix)
    shp_path = os.path.join(output_path, "HLNUG_queries")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(shp_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig = plt.figure(figsize=(10, 7), dpi=150)
    gs = fig.add_gridspec(ncols=2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1])
    axl.axis("off")
    shp_gdf.plot(ax=ax, fc="None", ec="C0", zorder=5)
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")

    with open(legend_path, "rb") as leg_file:
        uuid, label, style, facecolor, edgecolor, linewidth = pkl.load(
            leg_file
        )

    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    hydro_units_data = u4web.query_hlnug(
        "geologie/huek200/MapServer",
        "Hydrogeologische Einheiten",
        region=region,
        suffix=f"_{row[1].group:05}",
        out_folder=shp_path,
    )
    if len(hydro_units_data) > 0:
        fc = [
            (facecolor[uuid.index(ugrp)] if ugrp in uuid else (0.5, 0.5, 0.5))
            for ugrp in hydro_units_data["L_HE_B_KUE"].to_list()
        ]
        unugrp = np.unique(hydro_units_data["L_HE_B_KUE"].to_numpy())
        unugrp.sort()
        leg_handles = []
        leg_handles.append(
            mlines.Line2D([], [], color="C0", label="Bereich der Anomalie")
        )
        for ugrp in unugrp:
            try:
                ii = uuid.index(ugrp)
                facec = facecolor[ii]
            except ValueError:
                facec = (0.5, 0.5, 0.5)
            leg_handles.append(
                mpatches.Patch(fc=facec, label=textwrap.fill(label[ii], 50))
            )
        hydro_units_data.plot(ax=ax, fc=fc, ec="k", linewidth=0.25, alpha=0.5)
        axl.legend(
            handles=leg_handles,
            fontsize="small",
            markerscale=0.75,
        )
        u4ax.add_basemap(
            ax=ax,
            crs=hydro_units_data.crs,
            source=contextily.providers.CartoDB.Voyager,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_HUEK200.png")
        )
        fig.savefig(
            os.path.join(output_path, f"{row[1].group:05}_HUEK200.pdf")
        )
    plt.close(fig)


def make_topsoil_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    xlim: tuple,
    ylim: tuple,
    legend_path: os.PathLike,
):
    """Creates a map of topsoil units in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    :param legend_path: The path where the legend is found.
    :type legend_path: os.PathLike
    """
    output_path = os.path.join(output_path, suffix)
    shp_path = os.path.join(output_path, "HLNUG_queries")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(shp_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig = plt.figure(figsize=(10, 7), dpi=150)
    gs = fig.add_gridspec(ncols=2, width_ratios=(2, 1))
    ax = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1])
    axl.axis("off")
    shp_gdf.plot(ax=ax, fc="None", ec="C0", zorder=5)
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")

    with open(legend_path, "rb") as leg_file:
        uuid, label, style, facecolor, edgecolor, linewidth = pkl.load(
            leg_file
        )

    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    soil_data = u4web.query_hlnug(
        "boden/bfd50/MapServer",
        "BFD50_Bodenhauptgruppen",
        region=region,
        suffix=f"_{row[1].group:05}",
        out_folder=shp_path,
    )
    if len(soil_data) > 0:
        if "UNTERGRUPPE" in soil_data.keys():  # SHP files have less chars...
            column = "UNTERGRUPPE"
        else:
            column = "UNTERGRUPP"
        fc = [
            (facecolor[uuid.index(ugrp)] if ugrp in uuid else (0.5, 0.5, 0.5))
            for ugrp in soil_data[column].to_list()
        ]
        unugrp = np.unique(soil_data[column].to_numpy())
        unugrp.sort()
        leg_handles = []
        leg_handles.append(
            mlines.Line2D([], [], color="C0", label="Bereich der Anomalie")
        )
        for ugrp in unugrp:
            try:
                ii = uuid.index(ugrp)
                facec = facecolor[ii]
            except ValueError:
                facec = (0.5, 0.5, 0.5)
            leg_handles.append(
                mpatches.Patch(fc=facec, label=textwrap.fill(label[ii], 50))
            )
        soil_data.plot(ax=ax, fc=fc, ec="k", linewidth=0.25, alpha=0.5)
        axl.legend(
            handles=leg_handles,
            fontsize="small",
            markerscale=0.75,
        )
        u4ax.add_basemap(
            ax=ax,
            crs=soil_data.crs,
            source=contextily.providers.CartoDB.Voyager,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"{row[1].group:05}_BFD50.png"))
        fig.savefig(os.path.join(output_path, f"{row[1].group:05}_BFD50.pdf"))
    plt.close(fig)


def make_sat_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    xlim: tuple,
    ylim: tuple,
):
    """Creates a satellite overview of the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    """
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(dpi=150, figsize=(7, 7))
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")

    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    u4ax.add_basemap(
        ax=ax, crs=crs, source=contextily.providers.Esri.WorldImagery
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_satimg.png"))
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_satimg.pdf"))
    plt.close(fig)


def make_dem_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    dem_path: os.PathLike,
    xlim: tuple,
    ylim: tuple,
):
    """Creates a hillshade map of the digital elevation model in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param dem_path: The path where the dem data is found.
    :type dem_path: os.PathLike
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    """
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(dpi=150, figsize=(7, 7))
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")
    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    add_dem(ax, region, dem_path)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_dem.png"))
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_dem.pdf"))
    plt.close(fig)


def make_slope_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    dem_path: os.PathLike,
    xlim: tuple,
    ylim: tuple,
):
    """Creates a slope map of the digital elevation model in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param dem_path: The path where the dem data is found.
    :type dem_path: os.PathLike
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    """
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(dpi=150, figsize=(7, 7))
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")
    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    add_slope(ax, region, dem_path)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_slope.png"))
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_slope.pdf"))
    plt.close(fig)


def make_diffplan_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    diff_plan_path: os.PathLike,
    xlim: tuple,
    ylim: tuple,
):
    """Creates a map of the differences in surface elevation in the area of interest.

    :param row: The index and data of the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The path where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for plots.
    :type suffix: str
    :param diff_plan_path: The path where the differential data is found.
    :type diff_plan_path: os.PathLike
    :param xlim: The extend of the xaxis for consistent plotting.
    :type xlim: tuple
    :param ylim: The extend of the yaxis for consistent plotting.
    :type ylim: tuple
    """
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(dpi=150, figsize=(7, 7))
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")
    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    add_diff_plan(ax, region, diff_plan_path)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_diffplan.png"))
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_diffplan.pdf"))
    plt.close(fig)


def add_diff_plan(
    ax: mplax.Axes, region: gp.GeoDataFrame, tiff_folder: os.PathLike
):
    """Adds a differential motion plot to the axis.

    :param ax: The axis to plot into.
    :type ax: mplax.Axes
    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike
    """
    coverage = u4tiff.get_tiff_coverage(tiff_folder)
    if region.crs != coverage.crs:
        coverage = coverage.to_crs(region.crs)
    geom = region.geometry
    part = coverage.clip(geom)
    if len(part) > 0:
        for ii, fpath in enumerate(part.path.to_list()):
            if ii == 0:
                u4ax.add_tile(
                    fpath,
                    ax=ax,
                    vm=2,
                    cmap="RdBu",
                    colorbar={
                        "label": "Vertikaler Versatz (m)",
                        "shrink": 0.7,
                        "extend": "both",
                        # "pad": 0.05,
                    },
                )
            else:
                u4ax.add_tile(fpath, ax=ax, vm=2, cmap="RdBu")


def add_dem(ax: mplax.Axes, region: gp.GeoDataFrame, tiff_folder: os.PathLike):
    """Adds a digital elevation model dataset to the axis.

    :param ax: The axis to plot into.
    :type ax: mplax.Axes
    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(region, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    if len(file_list) > 0:
        for fpath in file_list:
            u4ax.add_tile(
                fpath, ax=ax, cmap="bone", hillshade=True, multidir=True
            )


def add_slope(
    ax: mplax.Axes, region: gp.GeoDataFrame, tiff_folder: os.PathLike
):
    """Adds a slope map to the axis.

    :param ax: The axis to plot into.
    :type ax: mplax.Axes
    :param region: The region where to extract the data.
    :type region: gp.GeoDataFrame
    :param tiff_folder: The folder where the data is found.
    :type tiff_folder: os.PathLike
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(region, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    if len(file_list) > 0:
        for ii, fpath in enumerate(file_list):
            if ii > 0:
                u4ax.add_tile(
                    fpath, ax=ax, cmap="inferno", slope=True, vm=(0, 45)
                )
            else:
                u4ax.add_tile(
                    fpath,
                    ax=ax,
                    cmap="inferno",
                    vm=(0, 45),
                    slope=True,
                    colorbar={
                        "label": "Hangneigung (°)",
                        "shrink": 0.7,
                        "extend": "both",
                        # "pad": 0.05,
                    },
                )


def make_detailed_map(
    row: tuple,
    crs: str,
    output_path: os.PathLike,
    suffix: str,
    hlnug_path: os.PathLike,
    contour_path: os.PathLike,
) -> Tuple[tuple, tuple]:
    """Makes a detailed overview map of the region including some geological features.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param crs: The coordinate system of the dataset.
    :type crs: str
    :param output_path: The folder where to store the output plots.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for saving.
    :type suffix: str
    :param hlnug_path: The path where the HLNUG data is found.
    :type hlnug_path: os.PathLike
    :param contour_path: The path where the contour dataset is found.
    :type contour_path: os.PathLike
    :return: The limits of the x and y axis for consistent plotting with other functions.
    :rtype: Tuple[tuple, tuple]
    """
    output_path = os.path.join(output_path, suffix)
    os.makedirs(output_path, exist_ok=True)

    shp_gdf = gp.GeoDataFrame(geometry=[row[1].geometry], crs=crs)
    fig, ax = plt.subplots(dpi=150, figsize=(7, 7))
    shp_gdf.plot(ax=ax, fc="None", ec="C0")
    shp_gdf.buffer(500).plot(ax=ax, fc="None", ec="None")

    u4pltfmt.map_style(ax=ax, divisor=500, crs=crs)
    fig.tight_layout()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    add_hlnug_data(
        ax,
        hlnug_path,
        "rutschungen_mittelpunkte_2021_06_21",
        color="k",
        marker="$\Swarrow$",
        markersize=30,
        label="Rutschungen, Mittelpunkte",
    )
    add_hlnug_data(
        ax,
        hlnug_path,
        "steinschlag_punkte",
        color="k",
        marker="$\therefore$",
        markersize=30,
        label="Steinschläge",
    )
    add_hlnug_data(
        ax,
        hlnug_path,
        "Erdfaelle_merged",
        color="k",
        marker="$\odot$",
        markersize=30,
        label="Erdfälle",
    )
    add_hlnug_data(
        ax,
        hlnug_path,
        "senkungsmulden",
        facecolor="None",
        edgecolor="k",
        linestyle=":",
        label="Senkungsmulden",
    )
    add_hlnug_data(
        ax,
        contour_path,
        "thresholded_contours_all_shapes",
        fc="None",
        column="color_levels",
    )
    ax.annotate(
        row[1].locations.replace(", ", "\n")[:-1],
        (0.99, 0.01),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
        zorder=10,
    )
    ax.legend(
        loc="upper right",
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    u4ax.add_basemap(
        ax=ax, crs=crs, source=contextily.providers.CartoDB.Voyager
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_map.png"))
    fig.savefig(os.path.join(output_path, f"{row[1].group:05}_map.pdf"))
    plt.close(fig)
    return (xlim, ylim)


def add_hlnug_data(
    ax: mplax.Axes, hlnug_path: os.PathLike, table: str, **plot_kwargs
):
    """Adds data from the HLNUG database to the plot.

    :param ax: The axis to plot into.
    :type ax: mplax.Axes
    :param hlnug_path: The path to the geodatabase with HLNUG data.
    :type hlnug_path: os.PathLike
    :param table: The sql table name to use.
    :type table: str
    """
    crs = u4sql.get_crs(hlnug_path, table)[0]
    region = gp.GeoDataFrame(
        geometry=[u4spatial.bounds_to_polygon(ax)], crs=crs
    )
    data = u4gpkg.load_gpkg_data_region_ogr(
        region, hlnug_path, table, clip=False
    )
    if len(data) > 0:
        data.plot(ax=ax, **plot_kwargs)


def make_tex_report(row: tuple, output_path:os.PathLike, suffix:str):
    """Creates a report for each area of interest using LaTeX. This is later merged together into a larger main document by the `main` function.

    :param row: The index and data for the area of interest.
    :type row: tuple
    :param output_path: The path where to store the outputs.
    :type output_path: os.PathLike
    :param suffix: The subfolder to use for the LaTeX files.
    :type suffix: str
    """
    output_path_tex = os.path.join(output_path, suffix)
    os.makedirs(output_path_tex, exist_ok=True)
    group = row[1].group
    img_path = os.path.join(output_path, "known_features", f"{group:05}")
    tex = (
        f"\\section{{Gruppe {group}}}\n\n"
        + f"\\textbf{{Lokalität:}} {row[1].locations}\n\n"
    )
    if os.path.exists(img_path + "_map.pdf") and os.path.exists(
        img_path + "_satimg.pdf"
    ):
        tex += (
            "\\begin{figure}[h!]\n"
            + "  \\centering\n"
            + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
            + f"    \\includegraphics[width=\\textwidth]{{{img_path+'_map.pdf'}}}\n"
            + "    \\caption{Übersicht über das Gebiet der Gruppe inklusive verschiedener Geogefahren und der detektierten Anomalien (Kartengrundlage OpenStreetMap).}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
            + f"    \\includegraphics[width=\\textwidth]{{{img_path+'_satimg.pdf'}}}\n"
            + "    \\caption{Luftbild basierend auf ESRI Imagery.}\n"
            + "  \\end{subfigure}\n"
            + "  \\caption{Lokalität der Anomalie.}"
            + "\\end{figure}\n\n"
        )

    # Differenzenplan
    tex += f"Im Gebiet um die detektierte Anomalie wurde insgesamt {row[1].volumes_moved}\\,m$^3$ Material bewegt, wovon {row[1].volumes_added}\\,m$^3$ hinzugefügt und {abs(row[1].volumes_removed)}\\,m$^3$ abgetragen wurde. Dies ergibt eine Gesamtbilanz von {row[1].volumes_total}\\,m$^3$, in Summe wurde also {vol_str(row[1].volumes_total)}.\n\n"
    if os.path.exists(img_path + "_diffplan.pdf") and os.path.exists(
        img_path + "_slope.pdf"
    ):
        tex += (
            "\\begin{figure}[!ht]\n"
            + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
            + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_diffplan.pdf'}}}\n"
            + "  \\caption{Differenzenplan im Gebiet.}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\begin{subfigure}[][][t]{.45\\textwidth}\n"
            + "\\centering\n"
            + f"  \\includegraphics[width=\\textwidth]{{{img_path+'_slope.pdf'}}}\n"
            + "  \\caption{Böschungswinkel im Gebiet der Gruppe.}\n"
            + "  \\end{subfigure}\n\hfill\n"
            + "  \\caption{Topographie im Gebiet.}"
            + "\\end{figure}\n\n"
        )

    # Topographie
    slope = float(row[1].slope_hull_median)
    slope_std = float(row[1].slope_hull_std)
    slope_rel = slope_std / slope
    landuse = ""
    try:
        landuse = eval(row[1].landuse_names)
    except NameError:
        landuse = row[1].landuse_names
    landuse_perc = eval(row[1].landuse_percent)
    tex += f"Die Steigung im Gebiet ist {slope_std_str(slope_std)} und {slope_str(slope)}. "

    if landuse:
        tex += (
            f"Der überwiegende Teil wird durch {landuse_str(row[1].landuse_major)} bedeckt. "
            + f"Die Anteile der Landnutzung sind: \n\n"
        )
        if isinstance(landuse, list):
            for ii in range(1, len(landuse) + 1):
                tex += f" {landuse_perc[-ii]:.1f}\\% {landuse_str(landuse[-ii])}, "
            tex = tex[:-2] + ".\n"
        else:
            tex += f"{landuse_perc:.1f}\\% {landuse_str(landuse)}"

    # Geogefahren
    tex += (
        "\\subsection*{Geogefahren}\n\n"
        + "\\begin{table}[H]\n"
        + "  \\centering"
        + "  \\caption{Bekannte Geogefahren}\n"
        + "  \\begin{tabular}{r|ll}\n"
        + "    Typ & Innerhalb des Areals & Im Umkreis von 1 km\\\\\\hline\n"
        + f"    Hangrutschungen & {row[1].landslides_num_inside} & {row[1].landslides_num_1km}\\\\\n"
        + f"    Karsterscheinungen & {row[1].karst_num_inside} & {row[1].karst_num_1km}\\\\\n"
        + f"    Steinschläge & {row[1].rockfall_num_inside} & {row[1].rockfall_num_1km}\n"
        + "  \\end{tabular}\n"
        + "\\end{table}\n\n"
    )

    lsar = row[1].landslide_total
    tex += "\\paragraph*{Rutschungsgefährdung}\n\n"
    if lsar > 0:
        tex += f"Das Gebiet liegt {part_str(lsar)} ({lsar:.0f}\%) in einem gefährdeten Bereich mit rutschungsanfälligen Schichten. "
        try:
            landslide_units = eval(row[1].landslide_units)
        except (NameError, SyntaxError):
            landslide_units = row[1].landslide_units
        if isinstance(landslide_units, list):
            tex += f"Die Einheiten sind: "
            for unit in landslide_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {landslide_units}.\n\n"
    else:
        tex += (
            "Es liegen keine Informationen zur Rutschungsgefährdung vor.\n\n"
        )

    ksar = row[1].karst_total
    tex += "\\paragraph*{Karstgefährdung}\n\n"
    if ksar > 0:
        tex += f"Das Gebiet liegt {part_str(ksar)} ({ksar:.0f}\%) in einem Bereich bekannter verkarsteter Schichten. "
        try:
            karst_units = eval(row[1].karst_units)
        except (NameError, SyntaxError):
            karst_units = row[1].karst_units
        if isinstance(karst_units, list):
            tex += f"Die Einheiten sind: "
            for unit in karst_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {karst_units}.\n\n"
    else:
        tex += "Es liegen keine Informationen zur Karstgefährdung vor.\n\n"

    subsar = row[1].subsidence_total
    tex += "\\paragraph*{Setzungsgefährdung}\n\n"
    if subsar > 0:
        tex += f"Das Gebiet liegt {part_str(subsar)} ({subsar:.0f}\%) in einem Bereich bekannter setzungsgefährdeter Schichten. "
        try:
            subsidence_units = eval(row[1].subsidence_units)
        except (NameError, SyntaxError):
            subsidence_units = row[1].subsidence_units
        if isinstance(subsidence_units, list):
            tex += f"Die Einheiten sind: "
            for unit in subsidence_units:
                tex += f"{unit}, "
            tex = tex[:-2] + ".\n\n"
        else:
            tex += f"Wichtigste Einheiten sind {subsidence_units}.\n\n"
    else:
        tex += "Es liegen keine Informationen zur Setzungsgefährdung vor.\n\n"

    # Geologie etc...
    tex += "\\subsection*{Geologie}\n\n"
    if os.path.exists(img_path + "_GK25.pdf"):
        tex += (
            "\\begin{figure}[H]\n"
            + "\\centering\n"
            + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_GK25.pdf'}}}\n"
            + "  \\caption{Geologie im Gebiet basierend auf GK25 (Quelle: HLNUG).}\n"
            + "\\end{figure}\n\n"
            + "\\subsection*{Hydrogeologie}\n\n"
        )
    if os.path.exists(img_path + "_HUEK200.pdf"):
        tex += (
            "\\begin{figure}[H]\n"
            + "\\centering\n"
            + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_HUEK200.pdf'}}}\n"
            + "  \\caption{Hydrogeologische Einheiten im Gebiet basierend auf HÜK200 (Quelle: HLNUG).}\n"
            + "\\end{figure}\n\n"
            + "\\subsection*{Bodengruppen}\n\n"
        )
    if os.path.exists(img_path + "_BFD50.pdf"):
        tex += (
            "\\begin{figure}[H]\n"
            + "\\centering\n"
            + f"  \\includegraphics[width=.9\\textwidth]{{{img_path+'_BFD50.pdf'}}}\n"
            + "  \\caption{Bodenhauptgruppen im Gebiet basierend auf der BFD50 (Quelle: HLNUG).}\n"
            + "\\end{figure}\n\n"
        )

    with open(
        os.path.join(output_path_tex, f"{group:05}_info.tex"),
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as tex_file:
        tex_file.write(tex)


def vol_str(val: float) -> str:
    """Converts a volume estimate into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val > 100:
        return "Material hinzugefügt"
    elif val < 100:
        return "Material abgetragen"
    else:
        return "das Gesamtvolumen nur wenig verändert"


def landuse_str(in_str: str) -> str:
    """Translates the landuse strings from OSM into German.

    :param in_str: The landuse text from `fclass`.
    :type in_str: str
    :return: The German translation.
    :rtype: str
    """
    convert = {
        "allotments": "Kleingärten",
        "buildings": "Gebäude",
        "cemetery": "Friedhof",
        "commercial": "Gewerbe",
        "farmland": "Ackerland",
        "farmyard": "Hof",
        "forest": "Wald",
        "grass": "Gras",
        "heath": "Heide",
        "industrial": "Industrie",
        "meadow": "Wiese",
        "military": "Sperrgebiet",
        "nature_reserve": "Naturschutzgebiet",
        "orchard": "Obstgarten",
        "park": "Park",
        "quarry": "Steinbruch",
        "recreation_ground": "Erholungsgebiet",
        "residential": "Wohngebiet",
        "retail": "Einzelhandel",
        "roads": "Straßen",
        "scrub": "Gestrüpp",
        "unclassified": "nicht klassifiziert",
        "water": "Gewässer",
        "vineyard": "Weinberg",
    }
    return convert[in_str]


def slope_str(val: float) -> str:
    """Converts a slope estimate into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val < 5:
        return "überwiegend flach"
    elif val < 10:
        return "leicht abschüssig"
    elif val < 30:
        return "steil"
    else:
        return "sehr steil"


def slope_std_str(val: float) -> str:
    """Converts a standard deviation of data into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val < 0.1:
        return "gleichmäßig"
    elif val < 0.5:
        return "etwas unregelmäßig"
    else:
        return "sehr variabel"


def part_str(val: float) -> str:
    """Gets a qualitative descriptor for the area.

    :param val: The area coverage
    :type val: float
    :return: A string describing the proportion of an area.
    :rtype: str
    """
    if val < 10:
        return "zu einem geringen Teil"
    elif val < 33:
        return "teilweise"
    elif val < 66:
        return "zu einem großen Teil"
    elif val < 90:
        return "zu einem überwiegenden Teil"
    else:
        return "quasi vollständig"


if __name__ == "__main__":
    main()
