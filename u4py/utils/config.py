"""
Default configuration for some scripts and modules.
"""
from __future__ import annotations

import logging
import os
import sys

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
        format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_shape_config() -> dict:
    """Returns the parameters for buffering OSM shapes.

    :return: The configuration.
    :rtype: dict
    """
    # Names of the shapes for the legend
    name = {
        "build": "Buildings",
        "landuse": "Quarries and Industrial",
        "construction": "Construction Sites",
        "railway": "Railways",
        "mainroads": "Main Roads",
        "minor_roads": "Minor Roads",
        "water": "Waterways",
        "lakes": "Lakes",
        "power": "Wind Turbines",
        "parking": "Parking Lots",
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
