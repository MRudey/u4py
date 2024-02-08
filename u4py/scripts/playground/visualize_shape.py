import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
from tqdm import tqdm

import u4py.io.shp as u4shp
import u4py.utils.config as u4config


def main():
    map_foto_path = Path(
        "~/Documents/ArcGIS/Places/MapFoto_Regions.shp"
    ).expanduser()
    osm_folder = r"D:\Projekte\Umwelt4\hesse_shp"
    output_folder = Path("~/Documents/ArcGIS/INSAR_plots").expanduser()
    rois = gp.read_file(map_foto_path)
    roi = rois[rois.Name == "Darmstadt"]
    region_name = roi.Name.to_string().replace(" ", "")
    shp_cfg = u4config.get_shape_config()
    shp_data = dict()
    for osm_type in tqdm(
        shp_cfg["shp_file"].keys(), desc="Getting Clipped Shapefiles"
    ):
        shp_data[osm_type], _ = u4shp.get_clipped_shapefile(
            os.path.join(
                osm_folder,
                shp_cfg["shp_file"][osm_type],
            ),
            roi,
            fclass=shp_cfg["fclass"][osm_type],
        )

    fig, ax = plt.subplots(figsize=(8.268, 11.693))
    _show_shp_data(shp_data, shp_cfg, ax=ax)
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_folder, "Map_Darmstadt.pdf"), bbox_inches="tight"
    )


def _show_shp_data(shp_data, shp_cfg, ax):
    shp_data["water"].plot(ax=ax, color="C0", zorder=1)
    shp_data["lakes"].plot(ax=ax, color="C0", zorder=1)

    shp_data["build"].plot(ax=ax, color="k", zorder=2)
    shp_data["railway"].plot(ax=ax, color="C1", zorder=3, linewidth=1)


if __name__ == "__main__":
    main()
