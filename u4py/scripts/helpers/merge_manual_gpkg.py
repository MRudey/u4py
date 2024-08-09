"""Merges the manual classification from one gpkg to the other"""

from pathlib import Path

import geopandas as gp
import numpy as np
from tqdm import tqdm


def main():
    file_path_a = Path(
        r"~\Documents\ArcGIS\Filtered_Classified_Shapes.gpkg"
    ).expanduser()

    file_path_b = Path(
        r"~\Documents\ArcGIS\Filtered_Classified_Shapes_final_class.gpkg"
    ).expanduser()

    file_path_merged = Path(
        r"~\Documents\ArcGIS\Filtered_Classified_Shapes_merged.gpkg"
    ).expanduser()

    gdf_a = gp.read_file(file_path_a)
    gdf_b = gp.read_file(file_path_b)

    merge_keys = [
        "manual_class_1",
        "manual_class_2",
        "manual_class_3",
        "manual_comment",
        "manual_group",
        "manual_known",
        "manual_research",
        "manual_unclear_1",
        "manual_unclear_2",
        "manual_unclear_3",
    ]
    for idx_b, grp in tqdm(enumerate(gdf_b.group.to_list()), total=len(gdf_b)):
        grp_a = gdf_a.group.to_numpy()
        idx_a = int(np.argwhere(grp_a == grp).squeeze())
        for mrgk in merge_keys:
            gdf_a.loc[idx_a, mrgk] = gdf_b[mrgk].iloc[idx_b]

    gdf_a.to_file(file_path_merged)


if __name__ == "__main__":
    main()
