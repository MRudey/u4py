"""
Contains types for better type hints in Python
"""

import os
from argparse import Namespace
from configparser import ConfigParser
from typing import TypedDict

import geopandas as gp


class U4PathsConfig(TypedDict):
    """Paths for the u4py project"""

    base_path: os.PathLike
    ext_path: os.PathLike
    places_path: os.PathLike
    output_path: os.PathLike
    psi_path: os.PathLike
    processing_path: os.PathLike
    diff_plan_path: os.PathLike
    u4projects_path: os.PathLike
    tektonik_path: os.PathLike
    bld_path: os.PathLike
    piloten_path: os.PathLike
    base_map_path: os.PathLike
    subsubregions_path: os.PathLike
    psivert_path: os.PathLike
    psiew_path: os.PathLike
    results_path: os.PathLike


class U4Config(TypedDict):
    """Options for u4py"""

    overwrite: bool
    use_filtered: bool
    use_parallel: bool
    use_online: bool
    use_internal: bool
    generate_plots: bool
    overwrite_plots: bool
    generate_document: bool
    single_report: bool
    is_hlnug: bool


class U4Metadata(TypedDict):
    """Some additional data needed for the scripts"""

    report_title: str
    report_subtitle: str
    report_suffix: str


class U4Project(ConfigParser):
    """A u4py project configuration"""

    paths: U4PathsConfig
    config: U4Config
    metadata: U4Metadata


class U4ResDict(TypedDict):
    """A dictionary with results of classifications"""

    additional_areas: float
    area: float
    aspect_hull_mean_14: list
    aspect_hull_mean_19: list
    aspect_hull_mean_21: list
    aspect_hull_median_14: list
    aspect_hull_median_19: list
    aspect_hull_median_21: list
    aspect_hull_std_14: list
    aspect_hull_std_19: list
    aspect_hull_std_21: list
    aspect_polygons_mean_14: list
    aspect_polygons_mean_19: list
    aspect_polygons_mean_21: list
    aspect_polygons_median_14: list
    aspect_polygons_median_19: list
    aspect_polygons_median_21: list
    aspect_polygons_std_14: list
    aspect_polygons_std_19: list
    aspect_polygons_std_21: list
    buildings_area: float
    buildings: gp.GeoDataFrame
    geology_area: list
    geology_percent: list
    geology_units: list
    geology_mapnum: list
    geology_mapname: list
    geometry: gp.GeoDataFrame
    group: float
    hydro_area: list
    hydro_percent: list
    hydro_units: list
    karst_area: list
    karst_num_1km: float
    karst_num_inside: float
    karst_percent: list
    karst_total: float
    karst_units: list
    landslide_area: list
    landslide_percent: list
    landslide_total: float
    landslide_units: list
    landslides_num_1km: float
    landslides_num_inside: float
    landuse_area: list
    landuse_major: str
    landuse_names: list
    landuse_percent: list
    landuse_total: float
    manual_class_1: str
    manual_class_2: str
    manual_class_3: str
    manual_comment: str
    manual_group: float
    manual_known: bool
    manual_research: bool
    manual_unclear_1: bool
    manual_unclear_2: bool
    manual_unclear_3: bool
    roads_has_motorway: bool
    roads_has_primary: bool
    roads_has_secondary: bool
    roads_motorway_names: list
    roads_primary_names: list
    roads_secondary_names: list
    roads_motorway_length: list
    roads_primary_length: list
    roads_secondary_length: list
    roads_main_area: float
    roads_main: gp.GeoDataFrame
    roads_minor_area: float
    roads_minor: gp.GeoDataFrame
    rockfall_num_1km: float
    rockfall_num_inside: float
    shape_ellipse_a: list
    shape_ellipse_b: list
    shape_ellipse_theta: list
    shape_flattening: list
    shape_roundness: list
    shape_width: list
    shape_breadth: list
    shape_aspect: list
    slope_hull_mean_14: list
    slope_hull_mean_19: list
    slope_hull_mean_21: list
    slope_hull_median_14: list
    slope_hull_median_19: list
    slope_hull_median_21: list
    slope_hull_std_14: list
    slope_hull_std_19: list
    slope_hull_std_21: list
    slope_polygons_mean_14: list
    slope_polygons_mean_19: list
    slope_polygons_mean_21: list
    slope_polygons_median_14: list
    slope_polygons_median_19: list
    slope_polygons_median_21: list
    slope_polygons_std_14: list
    slope_polygons_std_19: list
    slope_polygons_std_21: list
    subsidence_area: list
    subsidence_percent: list
    subsidence_total: float
    subsidence_units: list
    structural_region: list
    timeseries_annual_cosine: float
    timeseries_annual_max_amplitude: float
    timeseries_annual_max_time: float
    timeseries_annual_sine: float
    timeseries_linear: float
    timeseries_num_psi: float
    timeseries_offset: float
    timeseries_semiannual_cosine: float
    timeseries_semiannual_max_amplitude: float
    timeseries_semiannual_max_time: float
    timeseries_semiannual_sine: float
    topsoil_area: float
    topsoil_percent: float
    topsoil_units: float
    volumes_added: float
    volumes_moved: float
    volumes_removed: float
    volumes_total: float
    volumes_polygons_added: float
    volumes_polygons_moved: float
    volumes_polygons_removed: float
    volumes_polygons_total: float
    water_area: float
    well_number: int


class U4Namespace(Namespace):
    """Commandline arguments for U4Py"""

    input: os.PathLike
    overwrite: bool
    cpus: int
