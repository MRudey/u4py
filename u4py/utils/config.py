"""
Default configuration for some scripts and modules.
"""

from __future__ import annotations

import logging
import os
import sys

from u4py.utils.types import ShapeCfgDict

# Setting the cpu count for parallel processing to a convenient amount.
if os.cpu_count() < 2:
    ResourceWarning("Not enough processors for parallel processing.")
    cpu_count = 1
elif os.cpu_count() < 16:  # e.g. Laptop and Workstations
    cpu_count = os.cpu_count() - 2
else:  # Servers
    cpu_count = os.cpu_count() - 4

in_path = ""
log_level = logging.INFO


def start_logger():
    """
    Starts the logging process with the log level defined in `config.
    log_level` (defaults to `logging.INFO`).

    This function can be inserted at the beginning of a script to show the
    progress of processing or other info.
    """
    logging.basicConfig(
        format="[P%(process)d] %(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_shape_config() -> ShapeCfgDict:
    """Returns the parameters for buffering OSM shapes.

    :return: The configuration.
    :rtype: ShapeCfgDict
    """
    # Names of the shapes for the legend
    name = {
        "build": "Buildings",
        "pois_area": "Unfavorable POIs, (Sports Centres, Pools)",
        "landfill": "Landfills",
        "landuse": "Unfavorable Landuse",
        "construction": "Construction Sites",
        "railway": "Railways",
        "mainroads": "Main Roads",
        "minor_roads": "Minor Roads",
        "water": "Waterways",
        "lakes": "Lakes",
        "power": "Wind Turbines",
        "traffic": "Parking Lots,...",
        "transport": "Terminals, Airports,...",
    }

    # Name of the shapefile containing the original data
    shp_file = {
        "build": "gis_osm_buildings_a_free_1.shp",
        "pois_area": "gis_osm_pois_a_free_1.shp",
        "landfill": "lan-lan.shp",
        "landuse": "gis_osm_landuse_a_free_1.shp",
        "construction": "lan-con.shp",
        "railway": "gis_osm_railways_free_1.shp",
        "mainroads": "gis_osm_roads_free_1.shp",
        "minor_roads": "gis_osm_roads_free_1.shp",
        "water": "gis_osm_waterways_free_1.shp",
        "lakes": "gis_osm_water_a_free_1.shp",
        "power": "pow-gen_gen-win.shp",
        "traffic": "gis_osm_traffic_a_free_1.shp",
        "transport": "gis_osm_transport_a_free_1.shp",
    }

    # List of feature classes to extract from the file (empty=all)
    fclass = {
        "build": [],
        "pois_area": [
            "sports_centre",
            "trackpitch",
            "swimming_pool",
            "wastewater_plant",
            "golf_course",
            "camp_site",
            "playground",
        ],
        "landfill": [],
        "landuse": [
            "quarry",
            "construction",
            "industrial",
            "commercial",
            "farmyard",
        ],
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
        "traffic": [],
        "transport": [],
    }

    # Default buffer size around each feature is 10 meters.
    buffer_dist = {
        "build": 15,
        "pois_area": 25,
        "landfill": 50,
        "landuse": 50,
        "construction": 50,
        "railway": 25,
        "mainroads": 50,
        "minor_roads": 10,
        "water": 15,
        "lakes": 15,
        "power": 250,
        "traffic": 15,
        "transport": 15,
    }

    # Classification buffers
    classify_buffers = {
        "roads": 50,
        "railways": 50,
        "buildings": 50,
    }

    # Plotting Stuff
    colors = {
        "build": "dimgray",
        "pois_area": "dimgray",
        "landfill": "rosybrown",
        "landuse": "rosybrown",
        "construction": "rosybrown",
        "railway": "black",
        "mainroads": "orange",
        "minor_roads": "green",
        "water": "aqua",
        "lakes": "aqua",
        "power": "yellow",
        "traffic": "blue",
        "transport": "blue",
    }

    zorder = {
        "build": 2,
        "pois_area": 2,
        "landfill": 1,
        "landuse": 1,
        "construction": 1,
        "railway": 5,
        "mainroads": 4,
        "minor_roads": 3,
        "water": 3,
        "lakes": 3,
        "power": 3,
        "traffic": 3,
        "transport": 3,
    }

    shp_cfg = {
        "buffer_dist": buffer_dist,
        "classify_buffers": classify_buffers,
        "name": name,
        "shp_file": shp_file,
        "fclass": fclass,
        "colors": colors,
        "zorder": zorder,
    }

    return shp_cfg
