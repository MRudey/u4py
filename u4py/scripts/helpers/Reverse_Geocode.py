"""
Geolocates the centres of the filtered shapes in the classified shapes database. The results are saved into a csv file for easier access by other scripts.
"""

from typing import Tuple

import geopandas as gp
import numpy as np
import shapely as shp
from geographiclib.geodesic import Geodesic
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location
from geopy.point import Point
from tqdm import tqdm


def main():
    # Read Data
    class_shp_gdf = gp.read_file(
        "/home/rudolf/Documents/umwelt4/SelectedSites_May24/Classified_Shapes.gpkg"
    )

    gdf_filtered = filter_gdf(class_shp_gdf)
    locations = []
    loc_descript = []
    geolocator = Nominatim(
        user_agent="u4py email=rudolf@geo.tu-darmstadt.de", timeout=5
    )
    centroids = gdf_filtered.to_crs("EPSG:4326").geometry.centroid.to_list()
    for point in tqdm(centroids, "Reverse Geolocation"):
        location = reverse_geolocate(point, geolocator)
        qdict = get_nearest_query(location.raw["address"])
        nearest = geolocate(qdict, geolocator)
        if not nearest:
            nearest = location
        dist, direct = get_distance_direction(location, nearest)

        locations.append(location.address)
        loc_desc = describe_location(location.raw, nearest.raw, dist, direct)
        loc_descript.append(loc_desc)
    with open(
        "/home/rudolf/Documents/umwelt4/SelectedSites_May24/cached_locations.txt",
        "wt",
        encoding="utf-8",
    ) as cache:
        for loc, locd in zip(locations, loc_descript):
            cache.write(f"{loc}; {locd}\n")


def filter_gdf(in_gdf: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Filters the geodataframe according to some criteria.

    :param in_gdf: The input geodataframe.
    :type in_gdf: gp.GeoDataFrame
    :return: The filtered geodataframe.
    :rtype: gp.GeoDataFrame
    """
    lands_f = in_gdf.landslides_num_inside > 0
    karst_f = in_gdf.karst_num_inside > 0
    rockf_f = in_gdf.rockfall_num_inside > 0

    filter_all = np.logical_or(np.logical_or(lands_f, karst_f), rockf_f)
    out_gdf = in_gdf[filter_all]

    return out_gdf


def reverse_geolocate(point: shp.Point, geolocator: Nominatim) -> Location:
    """
    Reverse geolocates a point using the geolocator. Respects Nominatims rate
    limits and retries.

    :param point: _description_
    :type point: shp.Point
    :param geolocator: _description_
    :type geolocator: Nominatim
    :return: _description_
    :rtype: Location
    """
    lon = point.x
    lat = point.y
    locator = RateLimiter(
        geolocator.reverse(Point(lat, lon)),
        min_delay_seconds=1.5,
        max_retries=5,
    )
    return locator.func


def get_nearest_query(address: dict) -> dict:
    """
    Returns a query dictionary for the nearest inhabited area close to an
    address given by Nominatim.

    :param address: The address dictionary from a nominatim Location
    :type address: dict
    :return: The query
    :rtype: dict
    """

    if "isolated_dwelling" in address.keys():
        city = address["isolated_dwelling"]
    elif "hamlet" in address.keys():
        city = address["hamlet"]
    elif "village" in address.keys():
        city = address["village"]
    elif "town" in address.keys():
        city = address["town"]
    elif "city" in address.keys():
        city = address["city"]
    else:
        city = ""
    if city:
        query_dict = {
            "city": city,
            # "postalcode": address["postcode"],
            "country": address["country"],
        }
    else:
        query_dict = {
            "postalcode": address["postcode"],
            "country": address["country"],
        }
    return query_dict


def geolocate(query_dict: dict, geolocator: Nominatim) -> Location:
    """
    Geolocates the address in the query dictionary. Respects Nominatims rate
    limits and retries.

    :param query_dict: The dictionary for the Nominatim query.
    :type query_dict: dict
    :param geolocator: The geolocator for querying.
    :type geolocator: Nominatim
    :return: The location.
    :rtype: Location
    """
    location = RateLimiter(
        geolocator.geocode(
            query=query_dict,
            addressdetails=True,
            extratags=True,
            country_codes="de",
            featuretype="settlement",
            namedetails=True,
        ),
        min_delay_seconds=1.5,
        max_retries=5,
    )
    return location.func


def get_distance_direction(
    location_a: Location, location_b: Location
) -> Tuple[float, float]:
    """
    Gets the distance in metres and direction in degrees of `location_a` from
    `location_b`.

    :param location_a: The location to describe.
    :type location_a: Location
    :param location_b: The nearest bigger dwelling for reference.
    :type location_b: Location
    :return: The distance and direction.
    :rtype: Tuple[float, float]
    """
    gdf_a = gp.GeoDataFrame(
        geometry=[shp.Point(location_a.longitude, location_a.latitude)],
        crs="EPSG:4326",
    ).to_crs("EPSG:32632")
    gdf_b = gp.GeoDataFrame(
        geometry=[shp.Point(location_b.longitude, location_b.latitude)],
        crs="EPSG:4326",
    ).to_crs("EPSG:32632")

    dist = gdf_a.distance(gdf_b).to_numpy()[0]
    direction = Geodesic.WGS84.Inverse(
        location_b.latitude,
        location_b.longitude,
        location_a.latitude,
        location_a.longitude,
    )["azi1"]
    if direction < 0:
        direction += 360
    return dist, direction


def describe_location(
    feature: dict, nearest: dict, distance: float, direction: float
) -> str:
    """Reformats the address dictionary into a descriptive text.

    :param address: The address dictionary.
    :type address: dict
    :return: The description.
    :rtype: str
    """
    loc_str = address_to_text(feature, nearest)
    road = road_to_text(feature, nearest)
    dist = np.round(distance / 1000, 1)
    loc_dir = direction_to_text(direction)

    desc = (
        f"Die Anomalien befinden sich im Bereich {loc_str}. "
        + f"Im {loc_dir} ca. {dist} km vom Mittelpunkt {loc_str} entfernt,"
        + f" in der Nähe {road}. "
    )

    return desc


def address_to_text(feature: dict, nearest: dict) -> str:
    locs = {
        "municipality": "der Gemeinde",
        "city": "der Stadt",
        "town": "der städtischen Siedlung",
        "village": "des Dorfes",
        "city_district": "des Stadtbezirks",
        "district": "des Bezirks",
        "borough": "der Gemeinde",
        "suburb": "des Vororts",
        "subdivision": "der Unterabteilung",
        "hamlet": "des Weilers",
        "croft": "des Gehöfts",
        "isolated_dwelling": "der Einzelsiedlung",
    }

    for nn in locs:
        if nn in feature["address"].keys():
            return f"{locs[nn]} {feature['address'][nn]}"
        elif nn in nearest["address"].keys():
            return f"{locs[nn]} {nearest['address'][nn]}"


def road_to_text(feature: dict, nearest: dict) -> str:
    classes = {}
    types = {}
    if "road" in feature["address"].keys():
        return f"der Straße {feature['address']['road']}"
    elif "road" in nearest["address"].keys():
        return f"der Straße {nearest['address']['road']}"
    else:
        if feature["addresstype"] == "road":
            if "class" in feature.keys() and "type" in feature.keys():
                return f"{feature['type']} {feature['class']}"
            elif "class" in feature.keys():
                return f"{feature['class']}"
            elif "type" in feature.keys():
                return f"{feature['type']}"


def direction_to_text(direction: float, lang: str = "de") -> str:
    """Converts an azimut between 0 and 360 to ordinal directions.

    :param direction: The direction with 0 = North and 180 = South
    :type direction: float
    :param lang: The language of the text (supported values: "en", "de", "abbrev"), defaults to "de"
    :type lang: str, optional
    :return: The ordinal direction as a string
    :rtype: str
    """

    limits = np.arange(11.25, 360 + 22.25, 22.5)
    abbrevs = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
        "N",
    ]
    translate_abbrev = {
        "de": {
            "N": "Norden",
            "NNE": "Nordnordosten",
            "NE": "Nordosten",
            "ENE": "Ostnordosten",
            "E": "Osten",
            "ESE": "Ostsüdosten",
            "SE": "Südosten",
            "SSE": "Südsüdosten",
            "S": "Süden",
            "SSW": "Südsüdwesten",
            "SW": "Südwesten",
            "WSW": "Westsüdwesten",
            "W": "Westen",
            "WNW": "Westnordwesten",
            "NW": "Nordwesten",
            "NNW": "Nordnordwesten",
        },
        "en": {
            "N": "North",
            "NNE": "North-northeast",
            "NE": "Northeast",
            "ENE": "East-northeast",
            "E": "East",
            "ESE": "East-southeast",
            "SE": "Southeast",
            "SSE": "South-southeast",
            "S": "South",
            "SSW": "South-southwest",
            "SW": "Southwest",
            "WSW": "West-southwest",
            "W": "West",
            "WNW": "West-northwest",
            "NW": "Northwest",
            "NNW": "North-northwest",
        },
    }
    for ii in range(len(limits)):
        desc = abbrevs[ii]
        if direction <= limits[ii]:
            break
    if lang != "abbrev":
        desc = translate_abbrev[lang][desc]
    return desc


if __name__ == "__main__":
    main()
