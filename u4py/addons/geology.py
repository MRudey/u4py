"""
Contains functions to work with geological data.
"""
import geopandas as gp


def get_tektonik_hessen(tektonik_path, bld_path):
    tektonik = gp.read_file(tektonik_path).to_crs("EPSG:32632")
    bld = gp.read_file(bld_path).to_crs("EPSG:32632")
    hessen = bld[bld["GEN"] == "Hessen"]
    tektonik_hessen = gp.clip(tektonik, hessen)
    return tektonik_hessen
