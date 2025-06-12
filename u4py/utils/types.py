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

    overwrite_data: bool
    use_filtered: bool
    use_parallel: bool
    use_online: bool
    use_internal: bool
    generate_plots: bool
    overwrite_plots: bool
    overwrite: bool
    overwrite_reports: bool
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
    buildings_close: bool
    buildings_num: int
    buildings: gp.GeoDataFrame
    district: list
    geology_area: list
    geology_mapname: list
    geology_mapnum: list
    geology_percent: list
    geology_units: list
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
    railways_close: bool
    railways_has: bool
    railways_length: float
    roads_close: bool
    roads_has_motorway: bool
    roads_has_primary: bool
    roads_has_secondary: bool
    roads_has: bool
    roads_main_area: float
    roads_main: gp.GeoDataFrame
    roads_minor_area: float
    roads_minor: gp.GeoDataFrame
    roads_motorway_length: list
    roads_motorway_names: list
    roads_nearest_motorway_dist: float
    roads_nearest_motorway_name: str
    roads_nearest_primary_dist: float
    roads_nearest_primary_name: str
    roads_nearest_secondary_dist: float
    roads_nearest_secondary_name: str
    roads_primary_length: list
    roads_primary_names: list
    roads_secondary_length: list
    roads_secondary_names: list
    rockfall_num_1km: float
    rockfall_num_inside: float
    shape_aspect: list
    shape_breadth: list
    shape_ellipse_a: list
    shape_ellipse_b: list
    shape_ellipse_theta: list
    shape_flattening: list
    shape_roundness: list
    shape_width: list
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
    structural_region: list
    subsidence_area: list
    subsidence_percent: list
    subsidence_total: float
    subsidence_units: list
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
    volumes_error: float
    volumes_moved: float
    volumes_polygons_added: float
    volumes_polygons_moved: float
    volumes_polygons_removed: float
    volumes_polygons_total: float
    volumes_removed: float
    volumes_total: float
    water_area: float
    well_number: int


class U4Namespace(Namespace):
    """Commandline arguments for U4Py"""

    input: os.PathLike
    overwrite: bool
    cpus: int


class BufferDistDict(TypedDict):
    """Default buffer distances for features"""

    build: float
    pois_area: float
    landfill: float
    landuse: float
    construction: float
    railway: float
    mainroads: float
    minor_roads: float
    water: float
    lakes: float
    power: float
    traffic: float
    transport: float


class ClassifyDict(TypedDict):
    """Buffer distances for classification"""

    roads: float
    railways: float
    buildings: float


class NameDict(TypedDict):
    """Names for shapes of the legend"""

    build: str
    pois_area: str
    landfill: str
    landuse: str
    construction: str
    railway: str
    mainroads: str
    minor_roads: str
    water: str
    lakes: str
    power: str
    traffic: str
    transport: str


class ShpFileDict(TypedDict):
    """Names of the shapefile containing the original data"""

    build: os.PathLike
    pois_area: os.PathLike
    landfill: os.PathLike
    landuse: os.PathLike
    construction: os.PathLike
    railway: os.PathLike
    mainroads: os.PathLike
    minor_roads: os.PathLike
    water: os.PathLike
    lakes: os.PathLike
    power: os.PathLike
    traffic: os.PathLike
    transport: os.PathLike


class FclassDict(TypedDict):
    """List of feature classes to extract from the file"""

    build: list[str]
    pois_area: list[str]
    landfill: list[str]
    landuse: list[str]
    construction: list[str]
    railway: list[str]
    mainroads: list[str]
    minor_roads: list[str]
    water: list[str]
    lakes: list[str]
    power: list[str]
    traffic: list[str]
    transport: list[str]


class ColorsDict(TypedDict):
    """Colors for plotting features"""

    build: str
    pois_area: str
    landfill: str
    landuse: str
    construction: str
    railway: str
    mainroads: str
    minor_roads: str
    water: str
    lakes: str
    power: str
    traffic: str
    transport: str


class ZorderDict(TypedDict):
    """Layer `zorder` for plotting"""

    build: int
    pois_area: int
    landfill: int
    landuse: int
    construction: int
    railway: int
    mainroads: int
    minor_roads: int
    water: int
    lakes: int
    power: int
    traffic: int
    transport: int


class ShapeCfgDict(TypedDict):
    """A dictionary containing configs for shapes and buffers"""

    buffer_dist: BufferDistDict
    classify_buffers: ClassifyDict
    name: NameDict
    shp_file: ShpFileDict
    fclass: FclassDict
    colors: ColorsDict
    zorder: ZorderDict
