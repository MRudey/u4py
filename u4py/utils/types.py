"""
Contains types for better type hints in Python
"""

import datetime
import os
from argparse import Namespace
from configparser import ConfigParser
from typing import Optional, Tuple, TypedDict

import geopandas as gp
import numpy as np
import shapely


class U4PathsConfig(TypedDict):
    """Paths for the u4py project"""

    base_path: os.PathLike | str
    ext_path: os.PathLike | str
    places_path: os.PathLike | str
    output_path: os.PathLike | str
    psi_path: os.PathLike | str
    processing_path: os.PathLike | str
    diff_plan_path: os.PathLike | str
    u4projects_path: os.PathLike | str
    tektonik_path: os.PathLike | str
    bld_path: os.PathLike | str
    piloten_path: os.PathLike | str
    base_map_path: os.PathLike | str
    subsubregions_path: os.PathLike | str
    psivert_path: os.PathLike | str
    psiew_path: os.PathLike | str
    results_path: os.PathLike | str


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
    aspect_hull_mean_14: float | list
    aspect_hull_mean_19: float | list
    aspect_hull_mean_21: float | list
    aspect_hull_median_14: float | list
    aspect_hull_median_19: float | list
    aspect_hull_median_21: float | list
    aspect_hull_std_14: float | list
    aspect_hull_std_19: float | list
    aspect_hull_std_21: float | list
    aspect_polygons_mean_14: list | str
    aspect_polygons_mean_19: list | str
    aspect_polygons_mean_21: list | str
    aspect_polygons_median_14: list | str
    aspect_polygons_median_19: list | str
    aspect_polygons_median_21: list | str
    aspect_polygons_std_14: list | str
    aspect_polygons_std_19: list | str
    aspect_polygons_std_21: list | str
    buildings_area: float
    buildings_close: bool
    buildings_num: int
    buildings: gp.GeoDataFrame
    district: list | str
    geology_area: list | str
    geology_mapname: list | str
    geology_mapnum: list | str
    geology_percent: list | str
    geology_units: list | str
    geometry: shapely.Geometry
    group: int | str
    hydro_area: list | str
    hydro_percent: list | str
    hydro_units: list | str
    karst_area: list | str
    karst_num_1km: int
    karst_num_inside: int
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
    railways_length: list[float] | float
    roads_close: bool
    roads_has_motorway: bool
    roads_has_primary: bool
    roads_has_secondary: bool
    roads_has: bool
    roads_main_area: float
    roads_main: gp.GeoDataFrame
    roads_minor_area: float
    roads_minor: gp.GeoDataFrame
    roads_motorway_length: list[float] | float
    roads_motorway_names: list[str]
    roads_nearest_motorway_dist: float
    roads_nearest_motorway_name: str
    roads_nearest_primary_dist: float
    roads_nearest_primary_name: str
    roads_nearest_secondary_dist: float
    roads_nearest_secondary_name: str
    roads_primary_length: list[float] | float
    roads_primary_names: list[str]
    roads_secondary_length: list[float] | float
    roads_secondary_names: list[str]
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
    slope_hull_mean_14: list | float
    slope_hull_mean_19: list | float
    slope_hull_mean_21: list | float
    slope_hull_median_14: list | float
    slope_hull_median_19: list | float
    slope_hull_median_21: list | float
    slope_hull_std_14: list | float
    slope_hull_std_19: list | float
    slope_hull_std_21: list | float
    slope_polygons_mean_14: list | str
    slope_polygons_mean_19: list | str
    slope_polygons_mean_21: list | str
    slope_polygons_median_14: list | str
    slope_polygons_median_19: list | str
    slope_polygons_median_21: list | str
    slope_polygons_std_14: list | str
    slope_polygons_std_19: list | str
    slope_polygons_std_21: list | str
    structural_region: list | str
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
    topsoil_area: list[float]
    topsoil_percent: list[float]
    topsoil_units: list[str]
    volumes_added: float
    volumes_error: float
    volumes_moved: float
    volumes_polygons_added: float | list | str
    volumes_polygons_moved: float | list | str
    volumes_polygons_removed: float | list | str
    volumes_polygons_total: float | list | str
    volumes_removed: float
    volumes_total: float
    water_area: float
    well_number: int


class U4ResGdf(gp.GeoSeries):
    """A dictionary with results of classifications"""

    additional_areas: float
    aspect_hull_mean_14: float | list
    aspect_hull_mean_19: float | list
    aspect_hull_mean_21: float | list
    aspect_hull_median_14: float | list
    aspect_hull_median_19: float | list
    aspect_hull_median_21: float | list
    aspect_hull_std_14: float | list
    aspect_hull_std_19: float | list
    aspect_hull_std_21: float | list
    aspect_polygons_mean_14: list | str
    aspect_polygons_mean_19: list | str
    aspect_polygons_mean_21: list | str
    aspect_polygons_median_14: list | str
    aspect_polygons_median_19: list | str
    aspect_polygons_median_21: list | str
    aspect_polygons_std_14: list | str
    aspect_polygons_std_19: list | str
    aspect_polygons_std_21: list | str
    buildings_area: float
    buildings_close: bool
    buildings_num: int
    buildings: gp.GeoDataFrame
    district: list | str
    geology_area: list | str
    geology_mapname: list | str
    geology_mapnum: list | str
    geology_percent: list | str
    geology_units: list | str
    group: int | str
    hydro_area: list | str
    hydro_percent: list | str
    hydro_units: list | str
    karst_area: list | str
    karst_num_1km: int
    karst_num_inside: int
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
    locations: str
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
    roads_motorway_names: list | str
    roads_nearest_motorway_dist: float
    roads_nearest_motorway_name: str
    roads_nearest_primary_dist: float
    roads_nearest_primary_name: str
    roads_nearest_secondary_dist: float
    roads_nearest_secondary_name: str
    roads_primary_length: list | str
    roads_primary_names: list | str
    roads_secondary_length: list | str
    roads_secondary_names: list | str
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
    slope_hull_mean_14: list | float
    slope_hull_mean_19: list | float
    slope_hull_mean_21: list | float
    slope_hull_median_14: list | float
    slope_hull_median_19: list | float
    slope_hull_median_21: list | float
    slope_hull_std_14: list | float
    slope_hull_std_19: list | float
    slope_hull_std_21: list | float
    slope_polygons_mean_14: list | str
    slope_polygons_mean_19: list | str
    slope_polygons_mean_21: list | str
    slope_polygons_median_14: list | str
    slope_polygons_median_19: list | str
    slope_polygons_median_21: list | str
    slope_polygons_std_14: list | str
    slope_polygons_std_19: list | str
    slope_polygons_std_21: list | str
    structural_region: list | str
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
    volumes_polygons_added: float | list | str
    volumes_polygons_moved: float | list | str
    volumes_polygons_removed: float | list | str
    volumes_polygons_total: float | list | str
    volumes_removed: float
    volumes_total: float
    water_area: float
    well_number: int


class U4Namespace(Namespace):
    """Commandline arguments for U4Py"""

    input: os.PathLike | str
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

    build: os.PathLike | str
    pois_area: os.PathLike | str
    landfill: os.PathLike | str
    landuse: os.PathLike | str
    construction: os.PathLike | str
    railway: os.PathLike | str
    mainroads: os.PathLike | str
    minor_roads: os.PathLike | str
    water: os.PathLike | str
    lakes: os.PathLike | str
    power: os.PathLike | str
    traffic: os.PathLike | str
    transport: os.PathLike | str


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


class InversionResults(TypedDict):
    """Inversion results from u4py.analysis.inversion"""

    dhatE: np.ndarray
    dhatN: np.ndarray
    dhatU: np.ndarray
    dresE: np.ndarray
    dresN: np.ndarray
    dresU: np.ndarray


class InversionData(TypedDict):
    """Data structure used for the inversion algorithm"""

    t: np.ndarray
    dataE: np.ndarray
    dataN: np.ndarray
    dataU: np.ndarray
    sigmE: np.ndarray
    sigmN: np.ndarray
    sigmU: np.ndarray
    station: str | list[str]
    xmid: Optional[float | np.floating]
    ymid: Optional[float | np.floating]
    ori_inversion_results: Optional[np.ndarray]
    inversion_results: Optional[np.ndarray]
    ori_dhat_data: Optional[InversionResults]
    dhat_data: Optional[InversionResults]
    parameters_list: Optional[list]
    num_points: Optional[int]
    time: Optional[np.ndarray]
    x: Optional[np.ndarray]
    y: Optional[np.ndarray]
    ps_id: Optional[list]


class GfuncsDict(TypedDict):
    """Dictionary with individual G-function values"""

    lin: np.ndarray
    ann_sin: np.ndarray
    ann_cos: np.ndarray
    sem_sin: np.ndarray
    sem_cos: np.ndarray
    f_eq: np.ndarray
    f_at: np.ndarray
    f_ex: np.ndarray
    d_noise: np.ndarray


class GWStation(TypedDict):
    """Dictionary for groundwater data by HLNUG"""

    shortID: int
    gruwahID: int
    name: str
    easting: float
    northing: float
    time: list[datetime.datetime]
    height: list[float]


class WeatherStation(TypedDict):
    """Dictionary for weather data"""

    time: list[datetime.datetime]
    mean: list[float]
    max: list[float]
    min: list[float]
    rain: list[float]


class PegelLevelDict(TypedDict):
    """Dictionary with level data of river"""

    time: list[datetime.datetime]
    level: list[float]


class PegelTempDict(TypedDict):
    """Dictionary with temperature data of river"""

    time: list[datetime.datetime]


class PegelStation(TypedDict):
    """Dictionary for river level data"""

    name: str
    station: str
    water: str
    level: PegelLevelDict
    temperature: PegelTempDict


class EQStation(TypedDict):
    """Dictionary with earthquake data"""

    ID: int
    EVENT: str
    BUNDESLAND: str
    JAHR: int
    MONAT: int
    TAG: int
    ZEIT: str
    BREITE: float
    LAENGE: float
    LOKATION: str
    HERDTIEFE: float
    LOKALMAGNITUDE: float
    EPIZENTRALINTENSITAET: float
    REFERENZ_1: str
    BEMERKUNG: str
    MOMENTMAGNITUDE: float
    MAKROSEISMISCHE_MAGNITUDE: float
    MLMUS: float
    MLIH: float
    SR: str
    DATETIME: datetime.datetime
