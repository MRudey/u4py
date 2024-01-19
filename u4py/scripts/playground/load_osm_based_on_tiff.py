"""
Adds the bounds (for line/polygon) or xy coordinates for point features to the
sql table stored in a gpkg file.
"""
from pathlib import Path

import matplotlib.pyplot as plt

import u4py.plotting.axes as u4ax
import u4py.utils.config as u4config
import u4py.utils.files as u4files


def main():
    fpath = Path(
        r"~\Documents\ArcGIS\Places\OSM_shapes\all_shapes.gpkg"
    ).expanduser()
    tiff_path = r"D:\Projekte\Umwelt4\DGM-Differenzenplan_MitKorrektur\TB15\dgm1_32_487_5478_1_he_Kor_Diff.tif"
    shp_cfg = u4config.get_shape_config()

    buffered_gdf = u4files.load_osm_gpkg(fpath, tiff_path, shp_cfg=shp_cfg)

    # Needed only for plotting
    tiff_tile = u4files.load_tiff(tiff_path)

    fig, ax = plt.subplots()
    u4ax.add_tile(tiff_path, ax=ax, cmap="bone")
    buffered_gdf.plot(ax=ax, fc="None", column="fclass", legend=True)
    ax.set_xlim(tiff_tile.bounds.left, tiff_tile.bounds.right)
    ax.set_ylim(tiff_tile.bounds.bottom, tiff_tile.bounds.top)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
