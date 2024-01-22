"""
Testing Scipy Spatial functions to merge nearby polygons.
"""
import os
import pickle as pkl
from pathlib import Path

import geopandas as gp
import numpy as np
from tqdm import tqdm


def main():
    overwrite = False
    fpath = Path(
        r"~\Documents\ArcGIS\Places\thresholded_contours_all_shapes.gpkg"
    ).expanduser()
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
            x, y = pkl.read(pkl_file)

    close_shapes = "test"

    xx, yy = np.meshgrid(x, y)


if __name__ == "__main__":
    main()
