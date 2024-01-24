"""
Testing Scipy Spatial functions to merge nearby polygons.
"""
import os
import pickle as pkl
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt

import u4py.analysis.spatial as u4spatial


def main():
    overwrite = False
    # fpath = Path(
    #     r"~\Documents\ArcGIS\Places\thresholded_contours_all_shapes.gpkg"
    # ).expanduser()
    fpath = "/mnt/Raid/Umwelt4/hesse_shp/thresholded_contours_all_shapes.gpkg"
    folder, fname = os.path.split(fpath)
    pkl_path = os.path.join(folder, fname.replace(".gpkg", "_centroids.pkl"))
    if not os.path.exists(pkl_path) or overwrite:
        shapes = gp.read_file(fpath)
        x = shapes.centroid.x.to_numpy()
        y = shapes.centroid.y.to_numpy()
        with open(pkl_path, "wb") as pkl_file:
            pkl.dump((x, y), pkl_file)
    else:
        with open(pkl_path, "rb") as pkl_file:
            x, y = pkl.load(pkl_file)

    groups = u4spatial.group_nearest(x, y, 100)
    fig, ax = plt.subplots(dpi=150, figsize=(10, 10))
    ax.scatter(x, y, c=groups, cmap="rainbow")
    fig.tight_layout()
    fig.savefig("images/groups.png")


if __name__ == "__main__":
    main()
