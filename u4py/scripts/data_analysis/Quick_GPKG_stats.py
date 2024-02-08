"""
Some quick and dirty statistics for a gpkg file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import u4py.utils.sql as u4sql


def main():
    gpkg_path = Path(
        "~\Documents\ArcGIS\Data_2023\hessen_l3_clipped.gpkg"
    ).expanduser()

    print("Vertikal")

    velocities = np.array(
        u4sql.single_query(gpkg_path, "SELECT mean_velo_vert FROM vertikal")
    )

    print(f"mean: {np.mean(velocities):.2f}")
    print(f"std: {np.std(velocities):.2f}")
    print(f"median: {np.median(velocities):.2f}")
    print(
        f"median abs. dev.: {np.median(np.abs(velocities - np.median(velocities))):.2f}"
    )

    print("Ost - West")

    velocities = np.array(
        u4sql.single_query(gpkg_path, "SELECT mean_velo_east FROM Ost_West")
    )

    print(f"mean: {np.mean(velocities):.2f}")
    print(f"std: {np.std(velocities):.2f}")
    print(f"median: {np.median(velocities):.2f}")
    print(
        f"median abs. dev.: {np.median(np.abs(velocities - np.median(velocities))):.2f}"
    )


if __name__ == "__main__":
    main()
