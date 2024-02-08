"""
Testing various clustering algorithms to find Polygons that belong together
"""
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from tqdm import tqdm


def main():
    fpath = Path(
        r"~\Documents\ArcGIS\Places\thresholded_contours_all_shapes.gpkg"
    ).expanduser()
    shapes = gp.read_file(fpath)

    centroids = np.rot90(
        np.vstack(
            ([shapes.centroid.x.to_numpy()], [shapes.centroid.y.to_numpy()])
        )
    )

    results = dict()
    min_cluster_sizes = np.arange(2, 7)
    cluster_selection_epsilons = [50, 100, 150, 200]
    for min_cluster_size in tqdm(min_cluster_sizes, leave=False):
        for cluster_selection_epsilon in tqdm(
            cluster_selection_epsilons, leave=False
        ):
            db = HDBSCAN(
                min_cluster_size=min_cluster_size,
                cluster_selection_epsilon=cluster_selection_epsilon,
            ).fit(centroids)
            key = f"min_cluster_size{min_cluster_size}cluster_selection_epsilon{cluster_selection_epsilon}"
            results[key] = db.labels_

    gdf = gp.GeoDataFrame(geometry=shapes.geometry, data=results)
    gdf.to_file(
        Path(r"~\Documents\ArcGIS\Places\clustered_shapes.gpkg").expanduser()
    )


def print_and_plot(db, centroids):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise_ = list(db.labels_).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    unique_labels = set(db.labels_)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [
        plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
    ]
    fig, ax = plt.subplots()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = db.labels_ == k

        xy = centroids[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            ".",
            color=tuple(col),
        )

        xy = centroids[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            ".",
            color=tuple(col),
        )

    ax.set_title(f"Estimated number of clusters: {n_clusters_}")
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
