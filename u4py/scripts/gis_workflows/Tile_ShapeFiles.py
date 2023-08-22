import logging
import os

import geopandas as gp
import rasterio as rio
import shapely
import shapely.ops as shpops
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.utils.config as u4config
import u4py.utils.files as u4files
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    shp_cfg = get_shape_config()
    project = u4proj.get_project(required=["places_path", "diff_plan_path"])

    merged = get_merged_shapes(project, shp_cfg)
    tiff_file_list = u4files.get_file_list_tiff(
        project["paths"]["diff_plan_path"]
    )
    crs = rio.open(tiff_file_list[0]).crs
    merged = merged.to_crs(crs)
    tiff_coords = [
        u4spatial._bounds_to_coords(fpath)
        for fpath in tqdm(tiff_file_list, desc="Reading boundaries")
    ]
    tiff_polys = [
        shapely.Polygon([tfc[0], tfc[1], tfc[3], tfc[2], tfc[0]])
        for tfc in tqdm(tiff_coords, desc="Creating polygons")
    ]
    out_path = os.path.join(project["paths"]["places_path"], "cropped_shapes")
    os.makedirs(out_path, exist_ok=True)
    crop_shp_paths = [
        os.path.join(out_path, os.path.split(csp)[-1].replace(".tif", ".shp"))
        for csp in tiff_file_list
    ]
    for cshp, tifp in tqdm(
        zip(crop_shp_paths, tiff_polys),
        desc="Clipping shapes",
        total=len(crop_shp_paths),
    ):
        out = merged.clip(tifp)
        out.to_file(cshp)


def get_merged_shapes(project, shp_cfg):
    shp_data = dict()
    for osm_type in tqdm(
        shp_cfg["shp_file"].keys(),
        desc="Loading shapefiles",
        leave=False,
    ):
        # osm_type = "construction"
        shp_data[osm_type] = shp_load_buffer_union(
            os.path.join(
                project["paths"]["places_path"], shp_cfg["shp_file"][osm_type]
            ),
            shp_cfg["buffer_dist"][osm_type],
            shp_cfg["fclass"][osm_type],
        )

    logging.info("Adjusting CRS")
    for ii, kk in enumerate(shp_data.keys()):
        if ii == 0:
            first_crs = shp_data[kk].crs
        if shp_data[kk].crs != first_crs:
            shp_data[kk] = shp_data[kk].to_crs(first_crs)

    logging.info("Merging and buffering")
    out_gdf = gp.pd.concat([shp_data[kk] for kk in shp_data.keys()])
    out_gdf = out_gdf.to_crs()


def shp_load_buffer_union(
    shp_path: os.PathLike, buffer_dist: float, fclass: list
) -> gp.GeoDataFrame:
    geometries, crs = u4files.fiona_load(shp_path, fclass)
    logging.info("Buffering Geometry")
    buff_geom = [
        shapely.buffer(geom, buffer_dist) for geom in tqdm(geometries)
    ]
    logging.info("Returning GeoDataFrame")
    return gp.GeoDataFrame(geometry=buff_geom, crs=crs)


def get_shape_config():
    # Names of the shapes for the legend
    name = {
        "build": "Buildings",
        "landuse": "Steinbruch & Industrie",
        "construction": "Baustellen",
        "railway": "Railways",
        "mainroads": "Main Roads",
        "minor_roads": "Kleine Straßen",
        "water": "Waterways",
        "lakes": "Lakes",
        "power": "Windkraftanlagen",
        "parking": "Parkplätze",
    }
    # Name of the shapefile containing the original data
    shp_file = {
        "build": "gis_osm_buildings_a_free_1.shp",
        "landuse": "gis_osm_landuse_a_free_1.shp",
        "construction": "lan-con.shp",
        "railway": "gis_osm_railways_free_1.shp",
        "mainroads": "gis_osm_roads_free_1.shp",
        "minor_roads": "gis_osm_roads_free_1.shp",
        "water": "gis_osm_waterways_free_1.shp",
        "lakes": "gis_osm_water_a_free_1.shp",
        "power": "pow-gen_gen-win.shp",
        "parking": "par-sur.shp",
    }
    # List of feature classes to extract from the file (empty=all)
    fclass = {
        "build": [],
        "landuse": ["quarry", "construction", "industrial"],
        "construction": [],
        "railway": [],
        "mainroads": [
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
            "unclassified",  # necessary for some slip roads
        ],
        "minor_roads": [
            "residential",
            "living_street",
            "service",
            "pedestrian",
            "track",
            "track_grade5",
            "track_grade4",
            "track_grade3",
            "track_grade2",
            "track_grade1",
            "road",
            "bridleway",
            "steps",
            "path",
            "cycleway",
        ],
        "water": [],
        "lakes": [],
        "power": [],
        "parking": [],
    }
    # Default buffer size around each feature is 10 meters.
    buffer_dist = {
        "build": 10,
        "landuse": 10,
        "construction": 10,
        "railway": 15,
        "mainroads": 15,
        "minor_roads": 5,
        "water": 10,
        "lakes": 10,
        "power": 150,
        "parking": 10,
    }

    # Plotting Stuff
    colors = {
        "build": "dimgray",
        "landuse": "rosybrown",
        "construction": "rosybrown",
        "railway": "black",
        "mainroads": "orange",
        "minor_roads": "green",
        "water": "aqua",
        "lakes": "aqua",
        "power": "yellow",
        "parking": "blue",
    }
    zorder = {
        "build": 2,
        "landuse": 1,
        "construction": 1,
        "railway": 5,
        "mainroads": 4,
        "minor_roads": 3,
        "water": 3,
        "lakes": 3,
        "power": 3,
        "parking": 3,
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
