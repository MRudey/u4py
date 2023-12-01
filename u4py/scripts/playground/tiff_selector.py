"""
Interactive tiff selection for DEM/DSM correction
"""

import logging
import sys

import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import shapely
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4fmt
import u4py.utils.config as u4config
import u4py.utils.files as u4files

logging.basicConfig(
    format="[%(levelname)s] %(funcName)s: %(message)s",
    stream=sys.stdout,
    level=u4config.log_level,
)


def main():
    tiff_file_list = u4files.get_file_list(filetype=".tif")
    if tiff_file_list:
        selection = tiff_selection_plot(tiff_file_list)


def tiff_selection_plot(tiff_file_list):
    lims = {"left": 0, "right": 0, "bottom": 0, "top": 0}

    ALL_OK = False
    ADDING = True

    fig, ax = plt.subplots()
    dpi = fig.get_dpi()
    fig.set_size_inches((1920 / 2) / float(dpi), (1080 / 2) / float(dpi))
    plt.subplots_adjust(right=0.8)

    all_tiff_list = []
    for tiff_file in tqdm(tiff_file_list, desc="Reading tiffs"):
        bounds, crs = u4ax.add_tile(tiff_file, ax=ax, vm=0.3)
        all_tiff_list.append(u4spatial.bounds_to_polygon(bounds))
        lims = update_bounds(lims, bounds)
    all_tiff_gdf = gp.GeoDataFrame(
        {"source_path": tiff_file_list, "geometry": all_tiff_list}, crs=crs
    )
    SELECTION = []
    selection_plot = []
    all_tiff_gdf.plot(ax=ax, facecolor="C3", alpha=0.2)
    ax.set_xlim(lims["left"], lims["right"])
    ax.set_ylim(lims["bottom"], lims["top"])
    # u4fmt.map_style(ax, divisor=1000, crs=crs)

    def update_mode():
        nonlocal ADDING
        nonlocal ax
        if ADDING:
            ax.set_title("Adding Tiles.", color="C2")
        else:
            ax.set_title("Removing Tiles.", color="C3")

    update_mode()

    # Button callbacks
    def _cb_single(event):
        nonlocal ADDING
        nonlocal SELECTION
        nonlocal selection_plot
        nonlocal ax
        nonlocal crs
        nonlocal all_tiff_gdf
        logging.info("Single tile selection")
        sel = plt.ginput(1, show_clicks=True)
        pnt = shapely.Point(sel)
        sel_gdf = gp.GeoDataFrame({"geometry": [pnt]}, crs=crs)
        logging.info(f"Selection: {sel}")
        current_selection = gp.sjoin(sel_gdf, all_tiff_gdf, predicate="within")
        if ADDING:
            SELECTION.extend(list(current_selection.index_right))
        else:
            for ii in list(current_selection.index_right):
                try:
                    SELECTION.remove(ii)
                except ValueError:
                    pass
        update_selection(SELECTION, all_tiff_gdf, ax)

    def _cb_area(event):
        nonlocal ADDING
        nonlocal SELECTION
        logging.info("Area tile selection")

    def _cb_row(event):
        nonlocal ADDING
        nonlocal SELECTION
        logging.info("Row tile selection")

    def _cb_column(event):
        nonlocal ADDING
        nonlocal SELECTION
        logging.info("Column tile selection")

    def _cb_switch(event):
        nonlocal ADDING
        logging.info(f"Switch mode from {ADDING} to {not ADDING}")
        ADDING = not ADDING
        update_mode()

    def _cb_reset(event):
        logging.info("Reset selection")

    def _cb_save(event):
        logging.info("Save selection")
        nonlocal ALL_OK
        ALL_OK = not ALL_OK

    # Add Buttons
    btn_single = add_button(
        plt.axes([0.85, 0.8, 0.1, 0.05]),
        "Single",
        color="C0",
        hovercolor="C1",
        callback=_cb_single,
    )
    btn_area = add_button(
        plt.axes([0.85, 0.7, 0.1, 0.05]),
        "Area",
        color="C0",
        hovercolor="C1",
        callback=_cb_area,
    )
    btn_row = add_button(
        plt.axes([0.85, 0.6, 0.1, 0.05]),
        "Row",
        color="C0",
        hovercolor="C1",
        callback=_cb_row,
    )
    btn_column = add_button(
        plt.axes([0.85, 0.5, 0.1, 0.05]),
        "Column",
        color="C0",
        hovercolor="C1",
        callback=_cb_column,
    )
    btn_switch = add_button(
        plt.axes([0.85, 0.4, 0.1, 0.05]),
        "Add/Remove",
        color="C2",
        hovercolor="C1",
        callback=_cb_switch,
    )
    btn_reset = add_button(
        plt.axes([0.85, 0.3, 0.1, 0.05]),
        "Reset",
        color="C3",
        hovercolor="C1",
        callback=_cb_reset,
    )
    btn_save = add_button(
        plt.axes([0.85, 0.2, 0.1, 0.05]),
        "Save",
        color="C2",
        hovercolor="C1",
        callback=_cb_save,
    )

    # Keeps plot alive and stops further execution
    while not ALL_OK:
        if plt.get_fignums():
            plt.draw()
            plt.pause(0.01)
        else:
            sys.exit(0)


def update_selection(
    plot_indices: list,
    plot_gdb: gp.GeoDataFrame,
    ax: plt.Axes,
):
    while len(ax.collections) > 1:
        ax.collections[-1].remove()
        ax.collections.pop()
    plot_gdb.iloc[plot_indices].plot(color="C2", alpha=0.2, ax=ax)


def add_button(ax, label, color=None, hovercolor=None, callback=None):
    button = matplotlib.widgets.Button(
        ax, label, color=color, hovercolor=hovercolor
    )
    button.on_clicked(callback)
    return button


def update_bounds(bounds_old, bounds):
    if bounds_old["left"] == 0:
        bounds_old["left"] = bounds.left
    elif bounds_old["left"] > bounds.left:
        bounds_old["left"] = bounds.left

    if bounds_old["right"] == 0:
        bounds_old["right"] = bounds.right
    elif bounds_old["right"] < bounds.right:
        bounds_old["right"] = bounds.right

    if bounds_old["bottom"] == 0:
        bounds_old["bottom"] = bounds.bottom
    elif bounds_old["bottom"] > bounds.bottom:
        bounds_old["bottom"] = bounds.bottom

    if bounds_old["top"] == 0:
        bounds_old["top"] = bounds.top
    elif bounds_old["top"] < bounds.top:
        bounds_old["top"] = bounds.top

    return bounds_old


if __name__ == "__main__":
    main()
