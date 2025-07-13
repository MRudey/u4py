"""
Contains functions to work with geological data.
"""

from __future__ import annotations

import os

import geopandas as gp


def get_tektonik_hessen(
    tektonik_path: os.PathLike | str, bld_path: os.PathLike | str
) -> gp.GeoDataFrame:
    """Loads tectonic structures from a shapefile and clips them to Hesse.

    :param tektonik_path: Path to the tectonics shapefile.
    :type tektonik_path: os.PathLike | str
    :param bld_path: Path to the shapefile containing German states.
    :type bld_path: os.PathLike | str
    :return: GeoDataFrame with the clipped tectonic structures.
    :rtype: gp.GeoDataFrame
    """
    tektonik = gp.read_file(tektonik_path).to_crs("EPSG:32632")
    bld = gp.read_file(bld_path).to_crs("EPSG:32632")
    hessen = bld[bld["GEN"] == "Hessen"]
    tektonik_hessen = gp.clip(tektonik, hessen)
    return tektonik_hessen
