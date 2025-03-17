"""
Functions to query webservices, especially the arcgis rest api
"""

import io
import json
import logging
import os
import tempfile
from pathlib import Path

import geopandas as gp
import numpy as np
import PIL
import PIL.PngImagePlugin
import requests
import restapi
import shapely
from owslib.wfs import WebFeatureService
from owslib.wms import WebMapService, wms111, wms130

import u4py.analysis.spatial as u4spatial

# HLNUG_URL = "https://geodienste-umwelt.hessen.de/arcgis/rest/services"
HLNUG_URL = "https://geologie.hessen.de/arcgis/rest/services/"
INTERN_URL = "http://130.83.190.169:8080/geoserver/gk25-hessen/ows"


def query_hlnug(
    map_server_suffix: str,
    layer_name: str,
    region: shapely.Polygon = [],
    out_folder: os.PathLike = "",
    suffix: str = "",
    clip: bool = False,
) -> os.PathLike | gp.GeoDataFrame:
    """Queries the HLNUG Webservice for a specific layer.

    :param map_server_suffix: The suffix/subfolder on the server where to find the MapServer.
    :type map_server_suffix: str
    :param layer_name: The layer name to get the data from
    :type layer_name: str
    :param region: The region to search for data, defaults to [], getting the maximum data present (1000 entries max.).
    :type region: shapely.Polygon
    :param out_folder: The location where to store the data, defaults to "". If empty, uses a temporary folder and returns a GeoDataFrame instead of a path to a shapefile, defaults to ""
    :type out_folder: os.PathLike, optional
    :param layer_type: Define the type of the layer, defaults to "Feature Layer"
    :type layer_type: str, optional
    :param suffix: Suffix for identifying the intermediate savefile.
    :type suffix: str, optional
    :return: The data as geodataframe
    :rtype: gp.GeoDataFrame
    """
    # Save features first to json then to shapefile
    if not out_folder:
        with tempfile.TemporaryDirectory() as out_folder:
            out_fname = os.path.join(out_folder, f"{layer_name}{suffix}")
            features = _query_server(
                map_server_suffix, layer_name, region=region
            )
            if clip:
                gdf = _save_features(features, out_fname, region=region)
            else:
                gdf = _save_features(features, out_fname)
    else:
        os.makedirs(out_folder, exist_ok=True)
        out_fname = os.path.join(out_folder, f"{layer_name}{suffix}")
        if os.path.exists(out_fname + ".gpkg"):
            logging.info("Found locally cached file.")
            gdf = gp.read_file(out_fname + ".gpkg")

        else:
            features = _query_server(
                map_server_suffix, layer_name, region=region
            )
            if clip:
                gdf = _save_features(features, out_fname, region=region)
            else:
                gdf = _save_features(features, out_fname)
    return gdf


def query_internal(
    layer_name: str,
    region: shapely.Polygon = [],
    region_crs: str = "EPSG:32632",
    out_folder: os.PathLike = "",
    suffix: str = "",
    maxfeatures: int = 3000,
    login_path: os.PathLike = "~/Documents/umwelt4/login.json",
) -> gp.GeoDataFrame:
    """
    Reads the given layer from the internal Open WFS server (available only in
    the IAG).

    Requires a valid login!

    Available layers are:

        - `gk25-hessen:GK25_f-deck_GK3`
        - `gk25-hessen:GK25_f-dol_GK3`
        - `gk25-hessen:GK25_f-lin_GK3`
        - `gk25-hessen:GK25_f_GK3`
        - `gk25-hessen:GK25_gk-p_GK3`
        - `gk25-hessen:GK25_l-tek_GK3`
        - `gk25-hessen:GK25_p-deck_GK3`
        - `gk25-hessen:GK25_p-hydro_GK3`
        - `gk25-hessen:GK25_p-tek_GK3`

    :param layer_name: The name of the layer on the server. See list above.
    :type layer_name: str
    :param region: The region where to extract the data, defaults to []
    :type region: shapely.Polygon, optional
    :param region_crs: The crs of the region, defaults to "EPSG:32632"
    :type region_crs: str, optional
    :param out_folder: The location where to store the data, defaults to "". If empty, uses a temporary folder and returns a GeoDataFrame instead of a path to a shapefile, defaults to ""
    :type out_folder: os.PathLike, optional
    :param suffix: Suffix for identifying the intermediate savefile, defaults to ""
    :type suffix: str, optional
    :return: The data as geodataframe
    :rtype: gp.GeoDataFrame
    """

    # Save features first to json then to shapefile
    if not out_folder:
        with tempfile.TemporaryDirectory() as out_folder:
            out_fname = os.path.join(
                out_folder, f"{layer_name.replace(':', '_')}{suffix}"
            )
            features = _query_internal_server(
                layer_name, region, region_crs, login_path, maxfeatures
            )
            gdf = _save_features(features, out_fname)
    else:
        os.makedirs(out_folder, exist_ok=True)
        out_fname = os.path.join(
            out_folder, f"{layer_name.replace(':', '_')}{suffix}"
        )
        if os.path.exists(out_fname + ".gpkg"):
            logging.info("Found locally cached file.")
            gdf = gp.read_file(out_fname + ".gpkg")

        else:
            features = _query_internal_server(
                layer_name, region, region_crs, login_path, maxfeatures
            )
            gdf = _save_features(features, out_fname)
    return gdf


def _query_internal_server(
    layer_name: str,
    region: shapely.Polygon,
    region_crs: str,
    login_path: os.PathLike,
    maxfeatures: int,
) -> str:
    """Does the actual query of the internal IAG-GeoServer

    :param layer_name: The name of the layer on the server.
    :type layer_name: str
    :param region: The region where to extract the data
    :type region: shapely.Polygon
    :param region_crs: The crs of the region.
    :type region_crs: str
    :param login_path: The path to the login file.
    :type login_path: os.PathLike
    :param maxfeatures: The maximum amount of features to return
    :type maxfeatures: int
    :raises KeyError: Raised when the layer is not found on the server.
    :return: A JSON formatted reply from the webserver.
    :rtype: str
    """
    logging.info("Querying internal server for geology_data")
    login_path = Path(login_path).expanduser()
    if not os.path.exists(login_path):
        FileNotFoundError(
            "Provide a valid login file. A json file with the keys `user_name` and `password`."
        )

    with open(login_path, "rt") as loginfile:
        login = json.load(loginfile)
    try:
        intern_wfs = WebFeatureService(
            url=INTERN_URL,
            username=login["user_name"],
            password=login["password"],
            version="1.1.0",
        )
    except requests.exceptions.ConnectTimeout:
        logging.info("Connection to internal server timed out.")
        raise requests.exceptions.ConnectTimeout()
    lyr_names = list(intern_wfs.contents)
    if layer_name in lyr_names:
        # Check for correct crs:
        layer_crs = intern_wfs.contents[layer_name].crsOptions[0].id
        if region_crs != layer_crs:
            reg_gdf = gp.GeoDataFrame(
                geometry=[u4spatial.bounds_to_polygon(region)], crs=region_crs
            ).to_crs(layer_crs)
            bounds = reg_gdf.bounds.iloc[0]
            region = [bounds.minx, bounds.miny, bounds.maxx, bounds.maxy]

        response = intern_wfs.getfeature(
            typename=layer_name,
            bbox=region,
            srsname=intern_wfs.contents[layer_name].crsOptions[0],
            maxfeatures=maxfeatures,
            outputFormat="application/json",
        )
        response_data = json.loads(response.read())
    else:
        raise KeyError(f"Layer {layer_name} not found on Server: {lyr_names}")

    return response_data


def _save_features(
    features: str, out_fname: os.PathLike, region: gp.GeoDataFrame = []
) -> gp.GeoDataFrame:
    """Saves the features to geojson and shapefile.

    :param features: The features loaded from a query (in JSON format)
    :type features: str
    :param out_fname: The path where to store the data.
    :type out_fname: os.PathLike
    :return: A geodataframe with the features in EPSG:32632
    :rtype: gp.GeoDataFrame
    """
    if len(features) == 0:
        gdf = gp.GeoDataFrame(geometry=[])
    elif isinstance(features, dict):
        with open(out_fname + ".json", "wt") as geojson:
            json.dump(features, geojson)
        gdf = gp.read_file(out_fname + ".json").to_crs("EPSG:32632")
    else:
        with open(features.dump(out_fname + ".json")) as geojson:
            gdf = gp.read_file(geojson).to_crs("EPSG:32632")
    if len(region) > 0:
        gdf = gdf.clip(region)
    gdf.to_file(out_fname + ".gpkg")
    return gdf


def _query_server(
    map_server_suffix: str,
    layer_name: str,
    layer_type: str = "Feature Layer",
    region: shapely.Polygon = [],
) -> str:
    """Does the actual query of the Servers.

    :param map_server_suffix: The suffix of the folder on the webserver
    :type map_server_suffix: str
    :param layer_name: The name of the layer on the server.
    :type layer_name: str
    :param layer_type: The layer type of the layer., defaults to "Feature Layer"
    :type layer_type: str, optional
    :param region: The region where to extract the data., defaults to []
    :type region: shapely.Polygon, optional
    :raises TypeError: Raised when the layer is not of the correct layer type.
    :raises KeyError: Raised when the layer given is not found on the server.
    :return: A JSON formatted reply from the webserver.
    :rtype: str
    """
    # Query webservice to find layer
    logging.info("Querying HLNUG for geology_data")
    map_url = f"{HLNUG_URL}/{map_server_suffix}"
    feat_serv = restapi.MapService(
        map_url, client="u4py, rudolf@geo.tu-darmstadt.de"
    )
    lyr_types = [lyr.type for lyr in feat_serv.layers]
    lyr_names = [lyr.name for lyr in feat_serv.layers]

    if layer_name in lyr_names:
        ii = np.argwhere(layer_name == np.array(lyr_names)).squeeze()
        if lyr_types[ii] != layer_type:
            raise TypeError(
                f"Layer type mismatch: {lyr_types[ii]} != {layer_type}."
            )
    else:
        raise KeyError(f"Layer {layer_name} not found on Server: {lyr_names}")

    # Assemble url to layer and get data
    lyr_url = f"{map_url}/{ii}"
    ms_lyr = restapi.MapServiceLayer(
        lyr_url, client="u4py, rudolf@geo.tu-darmstadt.de"
    )
    if len(region) > 0:
        restgeom = polygon_to_restapi(
            region.geometry[0], region.crs.to_string()
        )
        try:
            features = ms_lyr.select_by_location(restgeom)
        except restapi.RestAPIException:
            features = []
    else:
        features = ms_lyr.query()
    return features


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


def wms_in_gdf_boundary(
    bound_gdf: gp.GeoDataFrame,
    wms_url: str,
    wms_version: str,
    layer: str,
    size: tuple,
) -> PIL.Image.Image:
    """Gets a WMS image within the region of a geodataframe.

    :param bound_gdf: The geodataframe to use for the region selection.
    :type bound_gdf: gp.GeoDataFrame
    :param wms_url: The URL of the WMS server
    :type wms_url: str
    :param wms_version: The version of the WMS server.
    :type wms_version: str
    :param layer: The layer to take from the WMS server
    :type layer: str
    :param size: The size of the image (maximum 3000x3000)
    :type size: tuple
    :raises ValueError: Raises a Value error if the layer is not available.
    :return: The image as a pillow image object for easy plotting with matplotlib.
    :rtype: PIL.Image.Image
    """
    wms = WebMapService(wms_url, version=wms_version)
    layers = list(wms.contents)

    # Check if the requested layer is there
    if layer in layers:
        lyr = wms[layer]
        # Get info from layer
        if hasattr(lyr, "crs_list"):
            lyr_crsopts = [
                v[-1] for v in lyr.crs_list if v[-1].startswith("EPSG:")
            ]  # use only a reduced set where the bounds are supplied
        elif hasattr(lyr, "crsOptions"):
            lyr_crsopts = [
                v[-1] for v in lyr.crsOptions if v[-1].startswith("EPSG:")
            ]
        else:
            raise AttributeError("Check the WMS layer crs attribute!")
        # Check if the input crs is supported
        if not str(bound_gdf.crs) in lyr_crsopts:
            # if not supported use default crs from layer
            bound_gdf = bound_gdf.to_crs(lyr_crsopts[0])

        # Check if the gdf lies within the supported range of the WMS
        if gdf_in_wms_bounds(lyr, bound_gdf):
            img = wms.getmap(
                layers=[layer],
                styles=["default"],
                srs=str(bound_gdf.crs),
                bbox=(
                    bound_gdf.bounds.minx[0],
                    bound_gdf.bounds.miny[0],
                    bound_gdf.bounds.maxx[0],
                    bound_gdf.bounds.maxy[0],
                ),
                size=size,
                format="image/png",
            )
            logging.info(img._response.url)
            img_dat = PIL.Image.open(io.BytesIO(img.read()))
        return img_dat

    else:
        raise ValueError(
            f"Layer {layer} not available for server. Valid layers are: {layers}."
        )


def gdf_in_wms_bounds(
    lyr: wms130.ContentMetadata | wms111.ContentMetadata,
    bound_gdf: gp.GeoDataFrame,
) -> bool:
    """Checks if the given geodataframe lies within the boundaries of the wms layer.

    :param lyr: The wms layer's meta data
    :type lyr: wms130.ContentMetadata | wms111.ContentMetadata
    :param bound_gdf: The geodataframe to use for wms queries.
    :type bound_gdf: gp.GeoDataFrame
    :raises ValueError: If boundaries are outside.
    :return: True when at least part of the data lies within WMS coverage.
    :rtype: bool
    """
    # Look for crs in respective lists and select boundary tuple
    crs_idx_list = [v[-1] for v in lyr.crs_list]
    crs_idx = crs_idx_list.index(str(bound_gdf.crs))
    lyr_bnd_box = lyr.crs_list[crs_idx]
    bnd_bnd_box = bound_gdf.bounds

    result = (
        # lies in x range
        (
            bnd_bnd_box.minx[0] >= lyr_bnd_box[0]
            and bnd_bnd_box.minx[0] <= lyr_bnd_box[2]
        )
        or (
            bnd_bnd_box.maxx[0] >= lyr_bnd_box[0]
            and bnd_bnd_box.maxx[0] <= lyr_bnd_box[2]
        )
    ) and (
        # lies in y range
        (
            bnd_bnd_box.miny[0] >= lyr_bnd_box[1]
            and bnd_bnd_box.miny[0] <= lyr_bnd_box[3]
        )
        or (
            bnd_bnd_box.maxy[0] >= lyr_bnd_box[1]
            and bnd_bnd_box.maxy[0] <= lyr_bnd_box[3]
        )
    )
    if result:
        return result
    else:
        raise ValueError(
            f"GeoDataFrame outside WMS boundaries. Please stay within {lyr_bnd_box}."
        )
