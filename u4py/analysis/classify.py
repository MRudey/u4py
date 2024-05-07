"""
Functions for the classification of a shape using the available datasets
"""

import configparser
import logging
import os

import geopandas as gp
import numpy as np
import shapely as shp

import u4py.addons.web_services as u4web
import u4py.analysis.inversion as u4invert
import u4py.analysis.other as u4other
import u4py.analysis.spatial as u4spatial
import u4py.io.gpkg as u4gpkg
import u4py.io.tiff as u4tiff
import u4py.plotting.plots as u4plots


def classify_shape(
    shp_gdf: gp.GeoDataFrame,
    group: int,
    buffer_size: float,
    shp_cfg: dict,
    project: dict,
    use_online: bool = False,
    save_shapes: bool = False,
    save_fig: bool = False,
    save_report: bool = False,
) -> dict:
    """Classifies a subset of shapes in `shp_gdf` based on the group. To
    include the surroundings of the shapes, a buffered hull around all shapes
    is used.

    :param shp_gdf: The loaded shapes to select the subset from.
    :type shp_gdf: gp.GeoDataFrame
    :param group: The name of the group.
    :type group: int
    :param buffer_size: The buffer size for the hull around the shapes.
    :type buffer_size: float
    :param shp_cfg: The configuration for shapes, e.g. including the buffer sizes for roads etc.
    :type shp_cfg: dict
    :param project: The project config containing file paths.
    :type project: dict
    :param use_online: Query online webservices by the HLNUG for geology, hydrogeology and soil, defaults to False
    :type use_online: bool, optional
    :param save_shapes: Save the individual shapes with results as single shape files, defaults to False
    :type save_shapes: bool, optional
    :param save_fig: Create some plots for each site., defaults to False
    :type save_fig: bool, optional
    :param save_report: Create a text report containing the results, defaults to False
    :type save_report: bool, optional
    :return: The results as a dictionary for further processing
    :rtype: dict

    As of now the following classifications are included:

        - **OSM Data**:

            - :py:func:`roads`: Does a motorway cross the area? Area of main and minor roads in the region.
            - :py:func:`buildings`: Area occupied by buildings.
            - :py:func:`rivers_water`: Area occupied by rivers and other water bodies.
            - :py:func:`landuse`: The land use in the area. Including roads, buildings and water if it has been computed before.

        - **Lidar**:

            - :py:func:`slope`: The average slope in the region and for each shape, computed on the basis of DEM 1m.
            - :py:func:`volume`: The volume of material moved based on the difference plans, split into different categories.

                - removed: sum of all negative displacements
                - added: sum of all positive displacements
                - moved: sum of all absolute displacements
                - total: sum of all displacements

        - **HLNUG Data**:

            - :py:func:`geology`: Area of all geological units in the region.
            - :py:func:`hydrogeology`: Area of all hydrological units in the region, including permeability.
            - :py:func:`topsoil`: Area of all topsoils in the region.
            - :py:func:`landslides`: Number of landslides in the region, in a buffer of 1 km around and the area of landslide prone layers in the region.
            - :py:func:`karst`: Number of karst phenomena in the region, in a buffer of 1 km around and the area of karst layers in the region.
            - :py:func:`rockfall`: Number of rockfalls in the region, in a buffer of 1 km around.
            - :py:func:`subsidence`: Number of subsidence phenomena in the region, in a buffer of 1 km around and the area of subsidence prone layers in the region.

        - **Others**:

            - :py:func:`shape`: Some shape parameters, e.g., roundness, ellipticity, for each polygon in the group.
    """
    osm_path = os.path.join(
        project["paths"]["places_path"], "OSM_shapes", "all_shapes.gpkg"
    )
    hlnug_path = os.path.join(
        project["paths"]["places_path"],
        "Classifier_shapes",
        "classifier_shapes.gpkg",
    )
    dem_path_new = os.path.join(
        project["paths"]["diff_plan_path"], "DGM1_2021", "raster_2021"
    )
    dem_path_old = os.path.join(
        project["paths"]["diff_plan_path"], "DGM1_2021", "raster_2021"
    )
    diffplan_path = os.path.join(
        project["paths"]["diff_plan_path"], "DGM-Differenzenplan_MitKorrektur"
    )
    psi_path = os.path.join(
        project["paths"]["psi_path"], "hessen_l3_clipped.gpkg"
    )

    # Getting hull for data extraction
    sub_set = u4spatial.get_subset(shp_gdf, group)
    if len(sub_set) > 0:
        sub_set_hull = u4spatial.get_subset_hull(shp_gdf, group, buffer_size)
    if len(sub_set_hull) > 0:
        # Classification
        res = preallocate_results()
        res["group"] = group
        res["geometry"] = sub_set_hull.geometry[0]
        res["area"] = round(res["geometry"].area, 1)
        res["additional_areas"] = 0

        # Manual Classification
        res.update(manual_classification(res, sub_set_hull, group, project))

        # Roads
        res.update(roads(res, sub_set_hull, osm_path, shp_cfg))

        # Buildings
        res.update(buildings(res, sub_set_hull, osm_path))

        # Water
        res.update(rivers_water(res, sub_set_hull, osm_path, shp_cfg))

        # Landuse
        res.update(landuse(res, sub_set_hull, osm_path))

        # Geometry
        res.update(
            slope(sub_set_hull, shp_gdf, dem_path_new, dem_path_old, group)
        )
        res.update(shape(sub_set))
        res.update(volume(sub_set_hull, diffplan_path))

        # Geology
        if use_online:
            res.update(geology(res, sub_set_hull))
            res.update(hydrogeology(res, sub_set_hull))
            res.update(topsoil(res, sub_set_hull))

        # Geohazard
        res.update(landslides(res, sub_set_hull, shp_path=hlnug_path))
        res.update(karst(res, sub_set_hull, shp_path=hlnug_path))
        res.update(rockfall(sub_set_hull, shp_path=hlnug_path))
        res.update(subsidence(res, sub_set_hull, shp_path=hlnug_path))

        # PSI
        res.update(psi_data(sub_set_hull, psi_path))

        if save_report:
            write_report(res, group, project)
        if save_shapes:
            write_shape(res, sub_set, group, project)
        if save_fig:
            write_fig(res, sub_set_hull, group, project)
        return res
    else:
        logging.info(f"No geometry found for group #{group}")
        return dict()


def preallocate_results() -> dict:
    """Preallocates all possible results for unified output and easier generation of master geodataframe.

    :return: A dictionary containing all keys but with empty values.
    :rtype: dict
    """
    res = {
        "additional_areas": np.nan,
        "area": np.nan,
        "buildings_area": np.nan,
        "buildings": gp.GeoDataFrame(),
        "geology_area": [],
        "geology_percent": [],
        "geology_units": [],
        "geometry": gp.GeoDataFrame(),
        "group": np.nan,
        "hydro_area": [],
        "hydro_percent": [],
        "hydro_units": [],
        "karst_area": [],
        "karst_num_1km": np.nan,
        "karst_num_inside": np.nan,
        "karst_percent": [],
        "karst_units": [],
        "karst_total": np.nan,
        "landslide_area": [],
        "landslide_percent": [],
        "landslide_total": np.nan,
        "landslide_units": [],
        "landslides_num_1km": np.nan,
        "landslides_num_inside": np.nan,
        "landuse_area": [],
        "landuse_names": [],
        "landuse_percent": [],
        "landuse_total": np.nan,
        "landuse_major": "",
        "manual_group": np.nan,
        "manual_known": False,
        "manual_research": False,
        "manual_class_1": "",
        "manual_unclear_1": False,
        "manual_class_2": "",
        "manual_unclear_2": False,
        "manual_class_3": "",
        "manual_unclear_3": False,
        "manual_comment": "",
        "roads_has_motorway": np.nan,
        "roads_main_area": np.nan,
        "roads_main": gp.GeoDataFrame(),
        "roads_minor_area": np.nan,
        "roads_minor": gp.GeoDataFrame(),
        "rockfall_num_1km": np.nan,
        "rockfall_num_inside": np.nan,
        "shape_ellipse_a": [],
        "shape_ellipse_b": [],
        "shape_ellipse_theta": [],
        "shape_flattening": [],
        "shape_roundness": [],
        "slope_hull_mean_old": [],
        "slope_hull_median_old": [],
        "slope_hull_std_old": [],
        "slope_polygons_mean_old": [],
        "slope_polygons_median_old": [],
        "slope_polygons_std_old": [],
        "slope_hull_mean_new": [],
        "slope_hull_median_new": [],
        "slope_hull_std_new": [],
        "slope_polygons_mean_new": [],
        "slope_polygons_median_new": [],
        "slope_polygons_std_new": [],
        "subsidence_area": [],
        "subsidence_percent": [],
        "subsidence_total": np.nan,
        "subsidence_units": [],
        "timeseries_annual_cosine": np.nan,
        "timeseries_annual_max_amplitude": np.nan,
        "timeseries_annual_max_time": np.nan,
        "timeseries_annual_sine": np.nan,
        "timeseries_linear": np.nan,
        "timeseries_num_psi": np.nan,
        "timeseries_offset": np.nan,
        "timeseries_semiannual_cosine": np.nan,
        "timeseries_semiannual_max_amplitude": np.nan,
        "timeseries_semiannual_max_time": np.nan,
        "timeseries_semiannual_sine": np.nan,
        "topsoil_area": np.nan,
        "topsoil_percent": np.nan,
        "topsoil_units": np.nan,
        "volumes_added": np.nan,
        "volumes_moved": np.nan,
        "volumes_removed": np.nan,
        "volumes_total": np.nan,
        "water_area": np.nan,
    }
    return res


def roads(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    osm_path: os.PathLike,
    shp_cfg: dict,
) -> dict:
    """Loads the roads from the openstreetmap database and returns the area of minor and major roads. Also establishes if a motorway is present in the area.

    :param res: The results dictionary as of now.
    :type res: dict
    :param sub_set_hull: The area where to look for roads.
    :type sub_set_hull: gp.GeoDataFrame
    :param osm_path: The path to the openstreetmap database.
    :type osm_path: os.PathLike
    :param shp_cfg: The shape config containing buffer sizes.
    :type shp_cfg: dict
    :return: An updated version of the results dictionary.
    :rtype: dict
    """

    def get_clipped_road_area(
        roads: gp.GeoDataFrame,
        road_type: str,
        clip_reg: gp.GeoDataFrame,
        shp_cfg: dict,
    ) -> gp.GeoDataFrame:
        """Buffers and clips the roads to the area of interest, removing tunnels as well.

        :param roads: The roads dataframe.
        :type roads: gp.GeoDataFrame
        :param road_type: The list of roads to use.
        :type road_type: str
        :param clip_reg: The region used for clipping.
        :type clip_reg: gp.GeoDataFrame
        :param shp_cfg: The shape config containing buffer sizes.
        :type shp_cfg: dict
        :return: The clipped roads as polygons.
        :rtype: gp.GeoDataFrame
        """
        roads = roads[roads["tunnel"] == "F"]
        road_gdf = gp.GeoDataFrame(
            geometry=gp.pd.concat(
                [
                    roads[roads["fclass"] == rds]
                    for rds in shp_cfg["fclass"][road_type]
                ]
            ).buffer(shp_cfg["buffer_dist"][road_type]),
            crs=roads.crs,
        )

        return road_gdf.clip(clip_reg)

    logging.info("Loading Road Data")
    roads_data = u4gpkg.load_gpkg_data_region_ogr(
        sub_set_hull, osm_path, "gis_osm_roads_free_1"
    )

    if len(roads_data) > 0:
        logging.info("Classifying Road Data")
        # Look for motorways
        res["roads_has_motorway"] = "motorway" in roads_data.fclass.to_list()

        # Extract area of main roads
        res["roads_main"] = get_clipped_road_area(
            roads_data, "mainroads", sub_set_hull, shp_cfg
        )
        if len(res["roads_main"]) > 0:
            res["roads_main_area"] = round(
                res["roads_main"].unary_union.area, 1
            )
            res["additional_areas"] += res["roads_main_area"]
        else:
            res["roads_main_area"] = 0

        # Extract area of minor roads
        res["roads_minor"] = get_clipped_road_area(
            roads_data, "minor_roads", sub_set_hull, shp_cfg
        )
        if len(res["roads_minor"]) > 0:
            res["roads_minor_area"] = round(
                res["roads_minor"].unary_union.area, 1
            )
            res["additional_areas"] += res["roads_minor_area"]

    return res


def buildings(
    res: dict, sub_set_hull: gp.GeoDataFrame, osm_path: os.PathLike
) -> dict:
    """Calculates the area that is covered by buildings

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param osm_path: The path to the osm dataset.
    :type osm_path: os.PathLike
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Loading Building Data")
    res["buildings"] = u4gpkg.load_gpkg_data_region_ogr(
        sub_set_hull, osm_path, "gis_osm_buildings_a_free_1"
    )
    if len(res["buildings"]) > 0:
        res["buildings_area"] = round(res["buildings"].area.sum(), 1)
        res["additional_areas"] += res["buildings_area"]
    else:
        res["buildings"] = []
    return res


def rivers_water(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    osm_path: os.PathLike,
    shp_cfg: dict,
) -> dict:
    """Calculates the area that is covered by water

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param osm_path: The path to the osm dataset.
    :type osm_path: os.PathLike
    :param shp_cfg: The shape config to calculate the buffers for the rivers.
    :type shp_cfg: dict
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Loading River Data")
    rivers_data = u4gpkg.load_gpkg_data_region_ogr(
        sub_set_hull, osm_path, "gis_osm_waterways_free_1"
    )
    if len(rivers_data) > 0:
        river_area = (
            rivers_data.buffer(shp_cfg["buffer_dist"]["water"])
            .clip(sub_set_hull)
            .unary_union.area
        )
    else:
        river_area = 0

    logging.info("Loading Water Data")
    water_data = u4gpkg.load_gpkg_data_region_ogr(
        sub_set_hull, osm_path, "gis_osm_water_a_free_1"
    )
    if len(water_data) > 0:
        water_area = water_data.area.sum()
    else:
        water_area = 0

    res["water_area"] = round(river_area + water_area, 1)
    res["additional_areas"] += res["water_area"]
    return res


def landuse(
    res: dict, sub_set_hull: gp.GeoDataFrame, osm_path: os.PathLike
) -> dict:
    """Calculates the area of each landuse including some other landuses from
    previous results, e.g. roads or water.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param osm_path: The path to the osm dataset.
    :type osm_path: os.PathLike
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Loading Landuse Data")
    landuse_data = u4gpkg.load_gpkg_data_region_ogr(
        sub_set_hull, osm_path, "gis_osm_landuse_a_free_1"
    )

    if len(landuse_data) > 0:
        logging.info("Classifying Landuse")
        # Remove landuse classes that might overlap and can be ignored
        fclasses = np.unique(landuse_data["fclass"])
        ignore_features = ["nature_reserve"]
        for ft in ignore_features:
            if ft in fclasses:
                landuse_data = landuse_data[landuse_data["fclass"] != ft]

    if len(landuse_data) > 0:
        # Remove areas that are covered by other features, e.g. roads
        if len(res["roads_main"]) > 0:
            landuse_data = landuse_data.overlay(
                res["roads_main"], how="difference"
            )
    if len(landuse_data) > 0:
        if len(res["roads_minor"]) > 0:
            landuse_data = landuse_data.overlay(
                res["roads_minor"], how="difference"
            )
    if len(landuse_data) > 0:
        if len(res["buildings"]) > 0:
            landuse_data = landuse_data.overlay(
                res["buildings"], how="difference"
            )

        # Calculate area
        res["landuse_names"], res["landuse_area"] = u4spatial.area_per_feature(
            landuse_data, "fclass"
        )

        # Add area of parts that were previously removed
        if len(res["roads_main"]) > 0 or len(res["roads_minor"]) > 0:
            if res["roads_main_area"] + res["roads_minor_area"] > 0:
                res["landuse_names"].append("roads")
                res["landuse_area"].append(
                    res["roads_main_area"] + res["roads_minor_area"]
                )
        if len(res["buildings"]) > 0:
            if res["buildings_area"] > 0:
                res["landuse_names"].append("buildings")
                res["landuse_area"].append(res["buildings_area"])
        if res["water_area"] > 0:
            res["landuse_names"].append("water")
            res["landuse_area"].append(res["water_area"])

        # Add rest as unclassified
        unclass_area = res["area"] - (
            landuse_data.area.sum() + res["additional_areas"]
        )
        if abs(unclass_area) > 1:
            res["landuse_names"].append("unclassified")
            res["landuse_area"].append(unclass_area)
            if unclass_area < 0:
                logging.info(f"Negative landuse for site {res['group']}!")
        # Sort landuse for better plots
        sorted = np.argsort(res["landuse_area"])
        res["landuse_names"] = list(np.array(res["landuse_names"])[sorted])
        res["landuse_area"] = list(
            np.round(np.array(res["landuse_area"])[sorted])
        )
        res["landuse_percent"] = [
            round((lnd / res["area"]) * 100, 1) for lnd in res["landuse_area"]
        ]
        res["landuse_major"] = res["landuse_names"][-1]
    res["additional_areas"] = round(res["additional_areas"], 1)
    return res


def slope(
    sub_set_hull: gp.GeoDataFrame,
    shp_gdf: gp.GeoDataFrame,
    dem_path_new: os.PathLike,
    dem_path_old: os.PathLike,
    group: str,
) -> dict:
    """Calculates the average slope in the area

    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param shp_gdf: The geodataframe including the polygons of the group.
    :type shp_gdf: gp.GeoDataFrame
    :param dem_path_new: The path to the dem folder of the new dem, after the events.
    :type dem_path_new: os.PathLike
    :param dem_path_old: The path to the dem folder of the old dem, before the events.
    :type dem_path_old: os.PathLike
    :param group: The group name
    :type group: str
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Calculating slope in all polygons")
    slope_polygons_new = u4spatial.calculate_slope_in_shapes(
        u4spatial.get_subset(shp_gdf, group), dem_path_new
    )
    slope_polygons_old = u4spatial.calculate_slope_in_shapes(
        u4spatial.get_subset(shp_gdf, group), dem_path_old
    )

    logging.info("Calculating average slope in hull")
    slope_hull_new = u4spatial.calculate_slope_in_shapes(
        sub_set_hull, dem_path_new
    )
    slope_hull_old = u4spatial.calculate_slope_in_shapes(
        sub_set_hull, dem_path_old
    )

    res = dict()
    # Old DEM results (before events)
    res["slope_polygons_mean_new"] = np.round(
        slope_polygons_new.slope_mean
    ).to_list()
    res["slope_polygons_median_new"] = np.round(
        slope_polygons_new.slope_median
    ).to_list()
    res["slope_polygons_std_new"] = np.round(
        slope_polygons_new.slope_std
    ).to_list()
    res["slope_hull_mean_new"] = np.round(slope_hull_new.slope_mean).to_list()
    res["slope_hull_median_new"] = np.round(
        slope_hull_new.slope_median
    ).to_list()
    res["slope_hull_std_new"] = np.round(slope_hull_new.slope_std).to_list()
    # New DEM results (after events)
    res["slope_polygons_mean_old"] = np.round(
        slope_polygons_old.slope_mean
    ).to_list()
    res["slope_polygons_median_old"] = np.round(
        slope_polygons_old.slope_median
    ).to_list()
    res["slope_polygons_std_old"] = np.round(
        slope_polygons_old.slope_std
    ).to_list()
    res["slope_hull_mean_old"] = np.round(slope_hull_old.slope_mean).to_list()
    res["slope_hull_median_old"] = np.round(
        slope_hull_old.slope_median
    ).to_list()
    res["slope_hull_std_old"] = np.round(slope_hull_old.slope_std).to_list()

    return res


def shape(sub_set: gp.GeoDataFrame) -> dict:
    """Calculates shape parameters for all shapes in the geodataframe.

    :param sub_set: The geodataframe with the polygons of the group.
    :type sub_set: gp.GeoDataFrame
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Computing shape parameters")
    roundness = u4spatial.roundness(sub_set)
    aa, bb, tt, flattn = u4spatial.flattening(sub_set)
    res = dict()
    res["shape_roundness"] = roundness
    res["shape_ellipse_a"] = aa
    res["shape_ellipse_b"] = bb
    res["shape_ellipse_theta"] = tt
    res["shape_flattening"] = flattn
    return res


def volume(sub_set_hull: gp.GeoDataFrame, diffplan_path: os.PathLike) -> dict:
    """Calculates the some volumetric quantities for the hull geometry.

    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param diffplan_path: The path to the folder where the diff plan tiffs are located.
    :type diffplan_path: os.PathLike
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Computing volumes in shapes")
    volumes = u4tiff.calculate_volume_in_shape(sub_set_hull, diffplan_path)
    res = dict()
    res["volumes_total"] = round(volumes.volume[0], 1)
    res["volumes_removed"] = round(volumes.volumes_removed[0], 1)
    res["volumes_added"] = round(volumes.volumes_added[0], 1)
    res["volumes_moved"] = round(volumes.volumes_moved[0], 1)
    return res


def geology(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = True,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the geological units and their spatial extend in the area.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to True
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data (not implemented yet), defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for geology_data")
        geology_data = u4web.query_hlnug(
            "geologie/gk25/MapServer",
            "Geologie (Kartiereinheiten)",
            region=sub_set_hull,
        ).clip(sub_set_hull)

        # Cut out parts where the data is None for some reason...
        unit_list = geology_data["GEOLOGISCHE_EINHEIT"].to_list()
        if None in unit_list:
            truth_array = [isinstance(unit, str) for unit in unit_list]
            geology_data = geology_data[truth_array]
    elif shp_path:
        logging.info("Loading geology_data from shapefile.")
        raise NotImplementedError()
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying Geology")
    res["geology_units"], res["geology_area"] = u4spatial.area_per_feature(
        geology_data, "GEOLOGISCHE_EINHEIT"
    )
    res["geology_percent"] = [
        round((area / res["area"]) * 100, 1) for area in res["geology_area"]
    ]
    return res


def hydrogeology(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = True,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the hydraulic conductivity and their spatial extend in the area.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to True
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data (not implemented yet), defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for hydro_units_data")
        hydro_units_data = u4web.query_hlnug(
            "geologie/huek200/MapServer",
            "Hydrogeologische Einheiten",
            region=sub_set_hull,
        ).clip(sub_set_hull)
    elif shp_path:
        logging.info("Loading hydro_units_data from shapefile.")
        raise NotImplementedError()
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying hydrogeology.")
    res["hydro_units"], res["hydro_area"] = u4spatial.area_per_feature(
        hydro_units_data, "L_CH_TXT"
    )
    res["hydro_percent"] = [
        round((area / res["area"]) * 100, 1) for area in res["hydro_area"]
    ]
    return res


def landslides(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = False,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the landslide prone units and their spatial extend in the area as
    well as the number of known landslides in the area.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to False
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data, defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for landslide_data")
        landslide_data = u4web.query_hlnug(
            "geologie/geogefahren/MapServer",
            "Rutschungsanfällige Schichten",
            region=sub_set_hull,
        ).clip(sub_set_hull)
        landslide_points = u4web.query_hlnug(
            "geologie/geogefahren/MapServer",
            "Rutschungsdatenbank",
            region=sub_set_hull.buffer(1000),
        )
    elif shp_path:
        logging.info("Loading landslide_data from shapefile.")
        landslide_data = u4gpkg.load_gpkg_data_region_ogr(
            sub_set_hull, shp_path, "rutschungsanfaellige_schichten_gk25"
        )
        landslide_points = u4gpkg.load_gpkg_data_region_ogr(
            sub_set_hull.buffer(1000),
            shp_path,
            "rutschungen_mittelpunkte_2021_06_21",
        )
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying landslides.")
    res["landslide_units"], res["landslide_area"] = u4spatial.area_per_feature(
        landslide_data, "GEN_TXT"
    )
    res["landslide_percent"] = [
        round((area / res["area"]) * 100, 1) for area in res["landslide_area"]
    ]
    res["landslide_total"] = sum(res["landslide_percent"])

    if len(landslide_points) > 0:
        res["landslides_num_1km"] = len(landslide_points)
        res["landslides_num_inside"] = len(landslide_points.clip(sub_set_hull))
    else:
        res["landslides_num_1km"] = 0
        res["landslides_num_inside"] = 0
    return res


def rockfall(
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = False,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the number of known rockfalls in the area.
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to False
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data, defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for rockfall_points")
        rockfall_points = u4web.query_hlnug(
            "geologie/geogefahren/MapServer",
            "Steinschlag- und Felssturzdatenbank",
            region=sub_set_hull.buffer(1000),
        )
    elif shp_path:
        logging.info("Loading rockfall_points from shapefile.")
        rockfall_points = u4gpkg.load_gpkg_data_region_ogr(
            sub_set_hull.buffer(1000), shp_path, "steinschlag_punkte"
        )
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying rockfall")
    res = dict()
    if len(rockfall_points) > 0:
        res["rockfall_num_1km"] = len(rockfall_points)
        res["rockfall_num_inside"] = len(rockfall_points.clip(sub_set_hull))
    else:
        res["rockfall_num_1km"] = 0
        res["rockfall_num_inside"] = 0
    return res


def subsidence(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = False,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the subsidence prone units and their spatial extend in the area.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to False
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data, defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for subsidence_data")
        subsidence_data = u4web.query_hlnug(
            "geologie/geogefahren/MapServer",
            "Setzungsempfindliche Schichten",
            region=sub_set_hull,
        ).clip(sub_set_hull)
    elif shp_path:
        logging.info("Loading subsidence_data from shapefile.")
        subsidence_data = u4gpkg.load_gpkg_data_region_ogr(
            sub_set_hull, shp_path, "setzungsempfindliche_schichten_gk25"
        )
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying subsidence")
    (
        res["subsidence_units"],
        res["subsidence_area"],
    ) = u4spatial.area_per_feature(subsidence_data, "GEN_TXT")
    res["subsidence_percent"] = [
        round((area / res["area"]) * 100, 1) for area in res["subsidence_area"]
    ]
    res["subsidence_total"] = sum(res["subsidence_percent"])
    return res


def karst(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = False,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the known karst risk and the spatial extend in the area as
    well as the number of known sinkholes in the area.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to False
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data, defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for karst_data")
        karst_data = u4web.query_hlnug(
            "geologie/geogefahren/MapServer",
            "Karstgefährdung",
            region=sub_set_hull,
        ).clip(sub_set_hull)
        karst_points = u4web.query_hlnug(
            "geologie/geogefahren/MapServer",
            "Erdfälle und Senkungsmulden",
            region=sub_set_hull.buffer(1000),
        )
    elif shp_path:
        logging.info("Loading karst_data from shapefile.")
        karst_data = u4gpkg.load_gpkg_data_region_ogr(
            sub_set_hull, shp_path, "karstgefaehrdung"
        )
        karst_points = u4gpkg.load_gpkg_data_region_ogr(
            sub_set_hull.buffer(1000), shp_path, "Erdfaelle_merged"
        )
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying karst.")
    res["karst_units"], res["karst_area"] = u4spatial.area_per_feature(
        karst_data, "KATEGORI"
    )
    res["karst_percent"] = [
        round((area / res["area"]) * 100, 1) for area in res["karst_area"]
    ]
    res["karst_total"] = sum(res["karst_percent"])
    if len(karst_points) > 0:
        res["karst_num_1km"] = len(karst_points)
        res["karst_num_inside"] = len(karst_points.clip(sub_set_hull))
    else:
        res["karst_num_1km"] = 0
        res["karst_num_inside"] = 0
    return res


def topsoil(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    use_online: bool = True,
    shp_path: os.PathLike = "",
) -> dict:
    """Gets the composition of the topsoil and its spatial extent in the area.

    :param res: The results dictionary including some previous results.
    :type res: dict
    :param sub_set_hull: The hull of the area.
    :type sub_set_hull: gp.GeoDataFrame
    :param use_online: Whether to query the online webservice of the HLNUG, defaults to True
    :type use_online: bool, optional
    :param shp_path: The path to a gpkg file containing the data (not implemented yet), defaults to ""
    :type shp_path: os.PathLike, optional
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    if use_online:
        logging.info("Querying HLNUG for topsoil_data")
        topsoil_data = u4web.query_hlnug(
            "boden/bfd50/MapServer",
            "BFD50_Bodenhauptgruppen",
            region=sub_set_hull,
        ).clip(sub_set_hull)
    elif shp_path:
        logging.info("Loading topsoil_data from shapefile.")
        raise NotImplementedError()
    else:
        raise ValueError("Please supply shapefile or set `use_online=True`.")

    logging.info("Classifying Topsoil")
    res["topsoil_units"], res["topsoil_area"] = u4spatial.area_per_feature(
        topsoil_data, "UNTERGRUPPE"
    )
    res["topsoil_percent"] = [
        round((area / res["area"]) * 100, 1) for area in res["topsoil_area"]
    ]
    return res


def psi_data(sub_set_hull: gp.GeoDataFrame, psi_path: os.PathLike) -> dict:
    """Gets the PSI Data in the region and does a time series inversion.

    :param sub_set_hull: The region to select the data.
    :type sub_set_hull: gp.GeoDataFrame
    :param psi_path: The path to the file containing the PSI data.
    :type psi_path: os.PathLike
    :return: The results dictionary with the classified data appended.
    :rtype: dict
    """
    logging.info("Loading PSI Data")
    data = u4gpkg.load_gpkg_data_region(sub_set_hull, psi_path, "vertikal")
    res = dict()
    if data:
        res["timeseries_num_psi"] = data["num_points"]
        logging.info("Inverting Data")
        params, _, _, _ = u4invert.invert_time_series(
            u4invert.reformat_dict(data)
        )

        logging.info("Parsing results")
        res["timeseries_offset"] = round(params[0], 2)
        res["timeseries_linear"] = round(params[1], 2)

        res["timeseries_annual_sine"] = round(params[2], 2)
        res["timeseries_annual_cosine"] = round(params[3], 2)
        max_vals, max_time = u4other.superpose(params[2], params[3])
        max_time = (max_time / (2 * np.pi)) * 365
        res["timeseries_annual_max_amplitude"] = round(max_vals, 2)
        res["timeseries_annual_max_time"] = round(max_time, 2)

        res["timeseries_semiannual_sine"] = round(params[4], 2)
        res["timeseries_semiannual_cosine"] = round(params[5], 2)
        max_vals, max_time = u4other.superpose(params[4], params[5])
        max_time = (max_time / (2 * np.pi)) * (365 / 2)
        res["timeseries_semiannual_max_amplitude"] = round(max_vals, 2)
        res["timeseries_semiannual_max_time"] = round(max_time, 2)
    return res


def write_shape(
    res: dict,
    sub_set: gp.GeoDataFrame,
    group: str,
    project: configparser.ConfigParser,
):
    """Writes the shapes including their individual results to a shapefile.

    :param res: The results dictionary.
    :type res: dict
    :param sub_set: The shapes for which the results where compiled.
    :type sub_set: gp.GeoDataFrame
    :param group: The name of the group.
    :type group: str
    :param project: The project containing paths.
    :type project: configparser.ConfigParser
    """
    logging.info("Creating folders")
    save_folder = os.path.join(project["paths"]["output_path"], "SiteShapes")
    os.makedirs(save_folder, exist_ok=True)
    shape_folder = os.path.join(save_folder, f"Site_{group}")
    os.makedirs(shape_folder, exist_ok=True)

    logging.info("Saving Data to shapefile")

    gdf = gp.GeoDataFrame(data=res, crs=sub_set.crs)
    gdf.to_file(os.path.join(shape_folder, f"{group}.shp"))


def write_report(res: dict, group: str, project: configparser.ConfigParser):
    """Saves the results for the specified group to a text file.

    :param res: The results dictionary.
    :type res: dict
    :param group: The name of the group.
    :type group: str
    :param project: The project containing paths.
    :type project: dict
    """
    logging.info("Writing report")
    report_folder = os.path.join(project["paths"]["output_path"], "Reports")
    os.makedirs(report_folder, exist_ok=True)
    with open(
        os.path.join(report_folder, f"Site_{group}.txt"),
        "wt",
        encoding="utf-8",
        newline="\n",
    ) as repf:
        repf.write(f"### Report File for Site {group}\n\n")
        for kk, vv in res.items():
            if kk != "geometry" and not isinstance(vv, gp.GeoDataFrame):
                repf.write(f"{kk} = {vv}\n")


def write_fig(
    res: dict,
    sub_set_hull: gp.GeoDataFrame,
    group: str,
    project: configparser.ConfigParser,
):
    """Creates several plots for each site, including statistics.

    :param res: The results dictionary.
    :type res: dict
    :param sub_set_hull: The hull of the region.
    :type sub_set_hull: gp.GeoDataFrame
    :param group: The name of the group.
    :type group: str
    :param project: The project containing paths.
    :type project: configparser.ConfigParser
    """
    logging.info("Creating Figures")
    save_folder = os.path.join(project["paths"]["output_path"], "SitePlots")
    os.makedirs(save_folder, exist_ok=True)
    u4plots.plot_shape(
        res["slope_polygons"],
        sub_set_hull,
        roads,
        res["slope_str"],
        group,
        save_folder=save_folder,
    )


def manual_classification(
    res: dict,
    sub_set_hull: shp.Polygon,
    group: str,
    project: configparser.ConfigParser,
) -> dict:
    """
    Loads the manually defined classification from a shape-file and checks if
    the hull overlaps with any of the points. This is necessary because in
    many cases the number of the group changes and only the spatial
    association is persisting.

    :param res: The results dictionary.
    :type res: dict
    :param sub_set_hull: The hull of the region.
    :type sub_set_hull: shp.Polygon
    :param group: The name of the group.
    :type group: str
    :param project: The project containing paths.
    :type project: configparser.ConfigParser
    :return: The results dictionary with the results appended.
    :rtype: dict
    """

    man_path = os.path.join(
        project["paths"]["sites_path"], "manual_classification.shp"
    )

    man_gdf = gp.read_file(man_path)
    interscts = man_gdf.intersects(sub_set_hull)
    if interscts.any():
        candidates = man_gdf[interscts]
        if len(candidates) > 1:
            logging.info(
                f"Multiple classifications found for group {group:05}"
            )
            for k in candidates:
                res[f"manual_{k}"] = candidates[k].to_list()
        else:
            for k in candidates:
                res[f"manual_{k}"] = candidates[k]
    else:
        logging.info(f"No manual classification found for group {group:05}")

    return res
