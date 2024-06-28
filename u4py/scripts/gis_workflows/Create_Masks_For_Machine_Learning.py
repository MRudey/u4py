"""
Creates masks to segment tiff files for machine-learning
"""

import configparser
import json
import os
from multiprocessing import Pool

import geopandas as gp
import numpy as np
import rasterio as rio
from rasterio import features as riofeat
from tqdm import tqdm

import u4py.io.gpkg as u4gpkg
import u4py.io.sql as u4sql
import u4py.io.tiff as u4tiff
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path="/home/rudolf/Documents/umwelt4/Create_Masks_For_Machine_Learning.u4project"
    )

    use_parallel = True

    shapes_path = os.path.join(
        project["paths"]["places_path"],
        "hesse_shp",
        "OSM_shapes",
        "all_shapes.gpkg",
    )
    anomaly_path = os.path.join(
        project["paths"]["results_path"], "Classified_Shapes.gpkg"
    )
    pilot_regionen = gp.read_file(
        os.path.join(project["paths"]["places_path"], "Pilotregionen.shp")
    )
    raster_folder = os.path.join(
        project["paths"]["diff_plan_path"], "DGM-Differenzenplan_MitKorrektur"
    )

    shp_cfg = u4config.get_shape_config()
    feature_index = create_feature_index(
        shp_cfg,
        shapes_path,
        anomaly_path,
        os.path.join(project["paths"]["output_path"], "feature_index.json"),
    )

    raster_list = u4tiff.get_tiff_regions(
        pilot_regionen[pilot_regionen.Name == "Rhein-Main"], raster_folder
    )
    os.makedirs(project["paths"]["output_path"], exist_ok=True)

    # Assembling arguments
    args = [
        (
            shapes_path,
            anomaly_path,
            raster_file,
            shp_cfg,
            feature_index,
            project,
        )
        for raster_file in raster_list
    ]
    if use_parallel:
        with Pool(u4config.cpu_count - 2) as p:
            list(
                tqdm(
                    p.imap_unordered(wrap_extract_masks, args),
                    total=len(args),
                    desc="Generating Plots",
                    leave=False,
                )
            )
    else:
        for arg in tqdm(
            args,
            total=len(args),
            desc="Generating Plots",
            leave=False,
        ):
            wrap_extract_masks(arg)


def get_masking_features(
    shapes_path: os.PathLike,
    anomaly_path: os.PathLike,
    raster_file: os.PathLike,
    shp_cfg: dict,
) -> gp.GeoDataFrame:
    """
    Combines the shapes in the area of the `raster_file` from `shapes_path`
    and `anomaly_path` into a single masking geodatabase. The shapes are
    buffered by their respective values in `shp_config`.

    :param shapes_path: The path to the osm shapefile.
    :type shapes_path: os.PathLike
    :param anomaly_path: The path to the classified anomaly shapefile.
    :type anomaly_path: os.PathLike
    :param raster_file: The path to the raster file.
    :type raster_file: os.PathLike
    :param shp_cfg: The shape configuration
    :type shp_cfg: dict
    :return: The merged shapes for masking.
    :rtype: gp.GeoDataFrame
    """
    features = u4gpkg.load_and_buffer_gpkg(
        shapes_path, raster_file, shp_cfg=shp_cfg
    )
    anomalies = u4gpkg.load_and_buffer_gpkg(anomaly_path, raster_file)
    if len(features) > 0 and len(anomalies) > 0:
        if features.crs != anomalies.crs:
            anomalies = anomalies.to_crs(features.crs)
        geometries = features.geometry.to_list()
        geometries.extend(anomalies.geometry.to_list())
        fclasses = [
            "osm_" + fl if fl else "osm_None"
            for fl in features.fclass.to_list()
        ]
        fclasses.extend(
            [
                "manclass_" + ano if ano else "manclass_None"
                for ano in anomalies.fclass.to_list()
            ]
        )
        masking_features = gp.GeoDataFrame(
            geometry=geometries,
            data={"fclass": fclasses},
            crs=features.crs,
        )
    elif len(features) > 0:
        fclasses = [
            "osm_" + fl if fl else "osm_None"
            for fl in features.fclass.to_list()
        ]
        masking_features = gp.GeoDataFrame(
            geometry=features.geometry,
            data={"fclass": fclasses},
            crs=features.crs,
        )
    elif len(anomalies) > 0:
        fclasses = [
            "manclass_" + ano if ano else "manclass_None"
            for ano in anomalies.fclass.to_list()
        ]
        masking_features = gp.GeoDataFrame(
            geometry=anomalies.geometry,
            data={"fclass": fclasses},
            crs=anomalies.crs,
        )
    else:
        masking_features = gp.GeoDataFrame()
    return masking_features


def raster_features(
    raster_file: os.PathLike,
    masking_features: gp.GeoDataFrame,
    feature_index: dict,
    project: configparser.ConfigParser,
):
    """
    Rasters the features in `masking_feature` using the `raster_file` as a
    template. The `feature_index` is used to translate the feature classes
    into a numeric value.

    :param raster_file: The path to the raster file.
    :type raster_file: os.PathLike
    :param masking_features: The masking features
    :type masking_features: gp.GeoDataFrame
    :param feature_index: The index translating feature class names into numbers.
    :type feature_index: dict
    :param project: The config containing the paths.
    :type project: configparser.ConfigParser
    """
    with rio.open(raster_file) as template:
        meta = template.meta.copy()
        meta.update(compress="lzw")
    _, fname_ext = os.path.split(raster_file)
    fname, ext = os.path.splitext(fname_ext)
    out_path = os.path.join(
        project["paths"]["output_path"], f"{fname}_mask{ext}"
    )
    with rio.open(out_path, "w+", **meta) as out:
        out_arr = out.read(1)
        mask_values = [
            feature_index[kk] for kk in masking_features.fclass.to_list()
        ]
        shapes = (
            (geom, value)
            for geom, value in zip(masking_features.geometry, mask_values)
        )
        burned = riofeat.rasterize(
            shapes, fill=0, out=out_arr, transform=out.transform
        )
        out.write_band(1, burned)


def create_feature_index(
    shp_cfg: dict,
    shapes_path: os.PathLike,
    anomaly_path: os.PathLike,
    out_path: os.PathLike,
) -> dict:
    """
    Creates an index to the features used for masking. Relates the value of
    the mask with the feature classes.

    :param shp_cfg: The shape config
    :type shp_cfg: dict
    :param shapes_path: The path to the osm shape file.
    :type shapes_path: os.PathLike
    :param anomaly_path: The path to the classified anomalies shape file.
    :type anomaly_path: os.PathLike
    :param out_path: The path to store the index.
    :type out_path: os.PathLike
    :return: A dictionary mapping feature names to values.
    :rtype: dict
    """
    feature_list = []
    for kk in shp_cfg["shp_file"].keys():
        if shp_cfg["fclass"][kk]:
            feature_list.extend(shp_cfg["fclass"][kk])
        else:
            feature_list.extend(
                u4sql.get_unique_entries(
                    shapes_path,
                    "fclass",
                    [shp_cfg["shp_file"][kk].replace(".shp", "")],
                )
            )
    feature_list = ["osm_" + fl for fl in feature_list]
    feature_list.append("osm_None")
    feature_list.append("osm_power")
    anomaly_list = np.unique(gp.read_file(anomaly_path)["manual_class_1"])
    anomaly_list = [
        "manclass_" + ano if ano else "manclass_None" for ano in anomaly_list
    ]
    feature_list.extend(anomaly_list)
    feature_index = dict()
    for ii, ft in enumerate(feature_list):
        feature_index[ft] = ii + 1
    with open(out_path, "wt") as jsfile:
        json.dump(feature_index, jsfile, indent=4)
    return feature_index


def wrap_extract_masks(args: tuple):
    """Parallel wrapper for `extract_masks`

    :param args: The function arguments
    :type args: tuple
    """
    extract_masks(*args)


def extract_masks(
    shapes_path, anomaly_path, raster_file, shp_cfg, feature_index, project
):
    masking_features = get_masking_features(
        shapes_path, anomaly_path, raster_file, shp_cfg
    )
    if len(masking_features) > 0:
        raster_features(raster_file, masking_features, feature_index, project)


if __name__ == "__main__":
    main()
