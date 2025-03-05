"""
Functions for converting numbers or other information to human readable text.
"""

from typing import Tuple

import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp


def slope_str(val: float) -> str:
    """Converts a slope estimate into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val < 1.1:
        return "nahezu eben"
    if val < 3.0:
        return "sehr leicht fallend"
    if val < 5.0:
        return "sanft geneigt"
    if val < 8.5:
        return "mäßig geneigt"
    if val < 16.5:
        return "stark ansteigend"
    if val < 24.0:
        return "sehr stark ansteigend"
    if val < 35.0:
        return "extrem ansteigend"
    if val < 45.0:
        return "steil"
    else:
        return "sehr steil"


def slope_std_str(val: float) -> str:
    """Converts a standard deviation of data into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val < 5:
        return "gleichmäßig"
    elif val < 10:
        return "etwas unregelmäßig"
    elif val < 15:
        return "unregelmäßig"
    else:
        return "sehr variabel"


def landuse_str(in_str: str) -> str:
    """Translates the landuse strings from OSM into German.

    :param in_str: The landuse text from `fclass`.
    :type in_str: str
    :return: The German translation.
    :rtype: str
    """
    convert = {
        "allotments": "Kleingärten",
        "buildings": "Gebäude",
        "cemetery": "Friedhof",
        "commercial": "Gewerbe",
        "farmland": "Ackerland",
        "farmyard": "Hof",
        "forest": "Wald",
        "grass": "Gras",
        "heath": "Heide",
        "industrial": "Industrie",
        "meadow": "Wiese",
        "military": "Sperrgebiet",
        "nature_reserve": "Naturschutzgebiet",
        "orchard": "Obstgarten",
        "park": "Park",
        "quarry": "Steinbruch",
        "recreation_ground": "Erholungsgebiet",
        "residential": "Wohngebiet",
        "retail": "Einzelhandel",
        "roads": "Straßen",
        "scrub": "Gestrüpp",
        "unclassified": "nicht klassifiziert",
        "water": "Gewässer",
        "vineyard": "Weinberg",
    }
    return convert[in_str]


def part_str(val: float) -> str:
    """Gets a qualitative descriptor for the area.

    :param val: The area coverage
    :type val: float
    :return: A string describing the proportion of an area.
    :rtype: str
    """
    if val < 10:
        return "zu einem geringen Teil"
    elif val < 33:
        return "teilweise"
    elif val < 66:
        return "zu einem großen Teil"
    elif val < 90:
        return "zu einem überwiegenden Teil"
    else:
        return "quasi vollständig"


def direction_to_text(
    direction: float | str, out_lang: str = "de", in_lang: str = "en"
) -> str:
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
    translate_abbrev_en = {  # Translate from English abbreviations
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
    translate_abbrev_de = {  # Translate from German abbreviations
        "de": {
            "N": "Norden",
            "NNO": "Nordnordosten",
            "NNW": "Nordnordwesten",
            "NO": "Nordosten",
            "NW": "Nordwesten",
            "O": "Osten",
            "ONO": "Ostnordosten",
            "OSO": "Ostsüdosten",
            "S": "Süden",
            "SO": "Südosten",
            "SSO": "Südsüdosten",
            "SSW": "Südsüdwesten",
            "SW": "Südwesten",
            "W": "Westen",
            "WNW": "Westnordwesten",
            "WSW": "Westsüdwesten",
        },
        "en": {
            "N": "North",
            "NNO": "North-northeast",
            "NNW": "North-northwest",
            "NO": "Northeast",
            "NW": "Northwest",
            "O": "East",
            "ONO": "East-northeast",
            "OSO": "East-southeast",
            "S": "South",
            "SO": "Southwest",
            "SSO": "South-southeast",
            "SSW": "South-southwest",
            "SW": "Southwest",
            "W": "West",
            "WNW": "West-northwest",
            "WSW": "West-southwest",
        },
    }

    if isinstance(direction, float):
        for ii in range(len(limits)):
            desc = abbrevs[ii]
            if direction <= limits[ii]:
                break
    elif isinstance(direction, str):
        desc = direction
    if out_lang != "abbrev":
        if in_lang == "en":
            desc = translate_abbrev_en[out_lang][desc]
        elif in_lang == "de":
            desc = translate_abbrev_de[out_lang][desc]
    return desc


def vol_str(val: float) -> str:
    """Converts a volume estimate into a descriptive text.

    :param val: The value.
    :type val: float
    :return: The descriptive text.
    :rtype: str
    """
    if val > 1000:
        return "Material hinzugefügt"
    elif val < 1000:
        return "Material abgetragen"
    else:
        return "das Gesamtvolumen nur wenig verändert"


def topo_text(
    means: float | list, std: float | list, unit: str, is_tex: bool
) -> Tuple[unc.ufloat, str]:
    """Converts the measured topography index, such as slope or aspect to a descriptive text.

    :param means: The mean of the value
    :type means: float | list
    :param std: The standard deviation of the value
    :type std: float | list
    """
    if isinstance(means, list):
        slope_unp = unp.uarray(
            means,
            std,
        )
        usl = np.mean(slope_unp)
    elif isinstance(means, float):
        usl = unc.ufloat(means, std)

    ustr = str(usl) + f"\\,{unit}"
    if "e" in ustr:
        ustr = f"{usl:.0f}\\,{unit}"

    if is_tex:
        ustr = ustr.replace("+/-", "$\\pm$")
    else:
        ustr = ustr.replace("+/-", "±")
        ustr = ustr.replace("\\,", "\u00a0")
    return usl, ustr


def listed_strings(entry: str | list) -> str:
    """
    Converts a possible list of strings into a more human-readable enumeration
    of strings.

    :param entry: A single string (returned immediately) or a list of strings to build the enumeration.
    :type entry: str | list
    :return: The list of strings with commas
    :rtype: str
    """

    # Sometimes we have strings that are lists??
    if isinstance(entry, str):
        if entry.startswith("["):
            entry = eval(entry)
        else:
            return entry

    if isinstance(entry, list):
        if len(entry) < 2:
            return entry[0]
        if len(entry) > 2:
            region_str = ""
            for ii, ent in enumerate(entry):
                if ii + 1 != len(entry):
                    region_str += f"{ent}, "
                else:
                    region_str = region_str[:-2] + f" und {entry}"
            return region_str
        else:
            return f"{entry[0]} und {entry[1]}"
    else:
        raise TypeError(f"Type of entry is invalid: {type(entry)}!")
