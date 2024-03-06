"""
Functions to query webservices, especially the arcgis rest api
    """

import os
import tempfile

import geopandas as gp
import numpy as np
import restapi
import shapely

HLNUG_URL = "https://geodienste-umwelt.hessen.de/arcgis/rest/services"


def query_hlnug(
    map_server_suffix: str,
    layer_name: str,
    region: shapely.Polygon = [],
    out_folder: os.PathLike = "",
    layer_type: str = "Feature Layer",
) -> os.PathLike | gp.GeoDataFrame:
    """Queries the HLNUG Webservice for a specific layer.

    :param map_server_suffix: The suffix/subfolder on the server where to find the MapServer.
    :type map_server_suffix: str
    :param layer_name: The layer name to get the data from
    :type layer_name: str
    :param region: The region to search for data, defaults to [], getting the maximum data present (1000 entries max.).
    :type region: shapely.Polygon
    :param out_folder: The location where to store the data, defaults to "". If empty, uses a temporary folder and returns a GeoDataFrame instead of a path to a shapefile.
    :type out_folder: os.PathLike
    :param layer_type: Define the type of the layer, defaults to "Feature Layer"
    :type layer_type: str, optional
    :return: The path to the saved shapefile.
    :rtype: os.PathLike | gp.GeoDataFrame
    """
    # Query webservice to find layer
    map_url = f"{HLNUG_URL}/{map_server_suffix}"
    feat_serv = restapi.MapService(map_url)
    lyr_types = [lyr.type for lyr in feat_serv.layers]
    lyr_names = [lyr.name for lyr in feat_serv.layers]

    if layer_name in lyr_names:
        ii = np.argwhere(layer_name == np.array(lyr_names)).squeeze()
        if lyr_types[ii] != layer_type:
            TypeError("Selected Layer is not a Feature Layer")
    else:
        KeyError("Layer not found on Server")

    # Assemble url to layer and get data
    lyr_url = f"{map_url}/{ii}"
    ms_lyr = restapi.MapServiceLayer(lyr_url)
    if len(region) > 0:
        restgeom = polygon_to_restapi(
            region.geometry[0], region.crs.to_string()
        )
        features = ms_lyr.select_by_location(restgeom)
    else:
        features = ms_lyr.query()

    # Save features first to json then to shapefile
    if not out_folder:
        with tempfile.TemporaryDirectory() as out_folder:
            out_fname = os.path.join(out_folder, layer_name)
            with open(features.dump(out_fname + ".json")) as geojson:
                gdf = gp.read_file(geojson).to_crs("EPSG:32632")
        return gdf
    else:
        out_fname = os.path.join(out_folder, layer_name)
        with open(features.dump(out_fname + ".json")) as geojson:
            gdf = gp.read_file(geojson).to_crs("EPSG:32632")
        gdf.to_file(out_fname + ".shp")
        return out_fname + ".shp"


def polygon_to_restapi(polygon: shapely.Polygon, crs: str) -> dict:
    """Converts a shapely polygon to a `restapi` compatible dictionary.

    :param polygon: The polygon.
    :type polygon: shapely.Polygon
    :param crs: The reference of the polygon, e.g. "EPSG:32632"
    :type crs: str
    :return: A dictionary in restapi compatible geoJSON.
    :rtype: dict
    """
    xx, yy = polygon.exterior.coords.xy
    geom = [
        [[round(x, 2), round(y, 2)] for x, y in zip(xx.tolist(), yy.tolist())]
    ]
    geometry = {
        "spatialReference": {"wkid": crs.replace("EPSG:", "")},
        "rings": geom,
    }

    return geometry
