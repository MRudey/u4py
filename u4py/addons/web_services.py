"""
Functions to query webservices, especially the arcgis rest api
    """

import os

import geopandas as gp
import restapi

HLNUG_URL = "https://geodienste-umwelt.hessen.de/arcgis/rest/services"


def hlnug_map_service_to_shapefile(
    map_server_suffix: str,
    layer_name: str,
    out_folder: os.PathLike,
    layer_type: str = "Feature Layer",
) -> os.PathLike:
    """Queries the HLNUG Webservice for a specific layer.

    :param map_server_suffix: The suffix/subfolder on the server where to find the MapServer.
    :type map_server_suffix: str
    :param layer_name: The layer name to get the data from
    :type layer_name: str
    :param out_folder: The location where to store the data.
    :type out_folder: os.PathLike
    :param layer_type: Define the type of the layer, defaults to "Feature Layer"
    :type layer_type: str, optional
    :return: The path to the saved shapefile.
    :rtype: os.PathLike
    """
    # Query webservice to find layer
    map_url = f"{HLNUG_URL}/{map_server_suffix}"
    feat_serv = restapi.MapService(map_url)
    for ii, lyr in enumerate(feat_serv.layers):
        if lyr.type == layer_type and lyr.name == layer_name:
            break

    # Assemble url to layer and get data
    lyr_url = f"{map_url}/{ii}"
    ms_lyr = restapi.MapServiceLayer(lyr_url)
    features = ms_lyr.query()

    # Save features first to json then to shapefile
    out_fname = os.path.join(out_folder, layer_name)
    with open(features.dump(out_fname + ".json")) as geojson:
        gdf = gp.read_file(geojson).to_crs("EPSG:25832")
    gdf.to_file(out_fname + ".shp")

    return out_fname + ".shp"
