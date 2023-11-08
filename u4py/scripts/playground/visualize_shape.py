import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
from tqdm import tqdm

import u4py.utils.files as u4files


def main():
    map_foto_path = Path(
        "~/Documents/ArcGIS/Places/MapFoto_Regions.shp"
    ).expanduser()
    osm_folder = r"D:\Projekte\Umwelt4\hesse_shp"
    output_folder = Path("~/Documents/ArcGIS/INSAR_plots").expanduser()
    rois = gp.read_file(map_foto_path)
    roi = rois[rois.Name == "Darmstadt"]
    region_name = roi.Name.to_string().replace(" ", "")
    shp_cfg = get_shape_config()
    shp_data = dict()
    for osm_type in tqdm(
        shp_cfg["shp_file"].keys(), desc="Getting Clipped Shapefiles"
    ):
        shp_data[osm_type], _ = u4files.get_clipped_shapefile(
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


def get_shape_config():
    # Names of the shapes for the legend
    name = {
        "build": "Buildings",
        "quarry": "Quarries",
        "railway": "Railways",
        "roads": "Roads",
        "water": "Waterways",
        "lakes": "Lakes",
    }
    # Name of the shapefile containing the original data
    shp_file = {
        "build": "gis_osm_buildings_a_free_1.shp",
        "quarry": "gis_osm_landuse_a_free_1.shp",
        "railway": "gis_osm_railways_free_1.shp",
        "roads": "gis_osm_roads_free_1.shp",
        "water": "gis_osm_waterways_free_1.shp",
        "lakes": "gis_osm_water_a_free_1.shp",
    }
    # List of feature classes to extract from the file (empty=all)
    fclass = {
        "build": [],
        "quarry": ["quarry"],
        "railway": [],
        "roads": [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
        ],
        "water": [],
        "lakes": [],
    }
    # Default buffer size around each feature is 10 meters.
    buffer_dist = {
        "build": 10,
        "quarry": 10,
        "railway": 10,
        "roads": 10,
        "water": 10,
        "lakes": 10,
    }

    # Plotting Stuff
    colors = {
        "build": "dimgray",
        "quarry": "rosybrown",
        "railway": "black",
        "roads": "orange",
        "water": "aqua",
        "lakes": "aqua",
    }
    zorder = {
        "build": 2,
        "quarry": 1,
        "railway": 5,
        "roads": 4,
        "water": 3,
        "lakes": 3,
    }

    shp_cfg = {
        "buffer_dist": buffer_dist,
        "name": name,
        "shp_file": shp_file,
        "fclass": fclass,
        "colors": colors,
        "zorder": zorder,
    }

    return shp_cfg


if __name__ == "__main__":
    main()
