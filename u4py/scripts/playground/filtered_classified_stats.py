import os
from pathlib import Path

import geopandas as gp
import matplotlib.axes as max
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder_path = Path(
        r"~\Documents\ArcGIS\SelectedSites_August24"
    ).expanduser()
    flist = [
        "Filtered_Classified_Shapes_hazard.gpkg",
        "Filtered_Classified_Shapes_onlyLarge.gpkg",
    ]
    for file in flist:
        plot_bar_diagram(folder_path, file)
    # fname_all = "Filtered_Classified_Shapes_all_merged_clean.gpkg"


def plot_bar_diagram(folder_path: os.PathLike | str, fname: str):
    gdf_large: gp.GeoDataFrame
    ax: max.Axes
    gdf_large = gp.read_file(os.path.join(folder_path, fname))

    fig, ax = plt.subplots()
    gdf_large.manual_class_1.value_counts(ascending=True).plot(
        ax=ax, kind="barh"
    )
    ax.annotate(
        f"Insgesamt {len(gdf_large)} Regionen,\ndavon {gdf_large.manual_research.value_counts(ascending=True).iloc[0]} genauer zu untersuchen\nund {gdf_large.manual_unclear_1.value_counts().iloc[1]} unklar.",
        (0.95, 0.05),
        xycoords="axes fraction",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.set_ylabel("")
    ax.set_xlabel("Anzahl")

    fig.tight_layout()
    outname = os.path.splitext(fname)[0].split("_")[-1]
    fig.savefig(os.path.join(folder_path, f"{outname}.pdf"))


if __name__ == "__main__":
    main()
