"""
Functions for handling tiff files.
"""

import logging
import os
from typing import Tuple

import geopandas as gp
import numpy as np
import rasterio as rio
import rasterio.mask as riomask
import scipy.stats as spstats
import shapely
from osgeo import gdal
from tqdm import tqdm


import u4py.analysis.processing as u4proc
import u4py.analysis.spatial as u4spatial
import u4py.io.files as u4files
import u4py.io.gpkg as u4gpkg
import u4py.utils.config as u4config


def batch_clip_tiff(args):
    """Wrapper for clipping with parallel Pool

    :param args: The arguments
    """
    clip_tiff_gpkg(*args)


def clip_tiff_gpkg(
    tiff_path: os.PathLike,
    gpkg_path: os.PathLike,
    ctiff_fol: os.PathLike,
):
    """Extracts the buffered shapes from the gpkg file and uses them to call `clip_tiff`.

    :param tiff_path: The path of the tif file to clip.
    :type tiff_path: os.PathLike
    :param gpkg_path: The path to the gpkg file containing all features for clipping.
    :type gpkg_path: os.PathLike
    :param ctiff_fol: The folder where to store the clipped tiffs.
    :type ctiff_fol: os.PathLike
    """
    try:
        folder, fname = os.path.split(tiff_path)
        _, folder = os.path.split(folder)
        shp_cfg = u4config.get_shape_config()
        shapes = u4gpkg.load_and_buffer_gpkg(
            gpkg_path, tiff_path, shp_cfg=shp_cfg
        )
        if len(shapes) > 0:
            clip_tiff(tiff_path, ctiff_fol, shapes.geometry.to_list())
    except ValueError:
        logging.info(f"Clipping of {folder}/{tiff_path} not successfull")


def clip_tiff(
    in_path: os.PathLike,
    ctiff_fol: os.PathLike,
    shapes: list,
    invert: bool = True,
):
    """Clips a tiff file with `shapes` and saves it to `ctiff_fol`.

    :param in_path: The path of the tif file to clip.
    :type in_path: os.PathLike
    :param ctiff_fol: The folder where to store the clipped tiffs.
    :type ctiff_fol: os.PathLike
    :param shapes: The shapes with which to clip
    :type shapes: list
    :param invert: Inverts the clip, if True data inside the shapes is clipped, defaults to True
    :type invert: bool, optional
    """
    logging.debug(f"Clipping {in_path}")
    _, fname = os.path.split(in_path)
    out_path = os.path.join(ctiff_fol, fname)
    with rio.open(in_path, "r") as src:
        out_image, out_transform = riomask.mask(src, shapes, invert=invert)
        out_meta = src.meta
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    with rio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)


def extract_xyz_tiff(
    file_path: os.PathLike,
) -> Tuple[dict, np.ndarray, str]:
    """Extracts the data and coordinates from a tiff file for easier processing with numpy.

    :param file_path: The path to the tiff file.
    :type file_path: os.PathLike
    :return: A Tuple with a dictionary containing the coordinates, the values as ndarray and a string with the coordinate system.
    :rtype: Tuple[dict, np.ndarray, str]
    """
    logging.debug("Reading data from tiff file.")
    with rio.open(file_path, "r") as tile:
        logging.debug("Loading tile data")
        zz = np.squeeze(tile.read())
        crs = tile.crs.to_string()
        shp = zz.shape

        logging.debug("Reading coordinates")
        x = []
        for ii in range(shp[1]):
            x.append(tile.xy(0, ii)[0])
        y = []
        for ii in range(shp[0]):
            y.append(tile.xy(ii, 0)[1])
    coords = {
        "x": np.min(x),
        "y": np.min(y),
        "dx": np.mean(np.diff(x)),
        "dy": np.mean(np.diff(y)),
    }
    return (coords, zz, crs)


def get_osm_tiff(
    query: dict, source_file_path: os.PathLike, overwrite: bool = False
) -> list[os.PathLike]:
    """Gets a list of paths to tiff files within a specified OSM query.

    :param query: The OSM query
    :type query: dict
    :param source_file_path: The path to the folder where the tiffs are stored (in `TB` folders)
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the output shapefile, defaults to False
    :type overwrite: bool, optional
    :return: A list of paths to the tiff files within the osm region.
    :rtype: list[os.PathLike]
    """
    file_list = u4files.get_file_list_tiff(source_file_path)
    points = u4spatial.select_points_osm(query, file_list)
    out_file_list = np.array(file_list)[points.source_ind]
    return np.unique(out_file_list).tolist()


def get_point_tiff(
    point: Tuple[float, float],
    radius: float,
    source_file_path: os.PathLike,
) -> list[os.PathLike]:
    """Returns paths of tiff files that are within `radius` of `point`.

    :param point: The point where to start.
    :type point: Tuple[float, float]
    :param radius: The search radius around point (in meters).
    :type radius: float
    :param source_file_path: The path to the folder where the tiffs are stored (in `TB` folders)
    :type source_file_path: os.PathLike
    :param overwrite: Whether to overwrite the output shapefile, defaults to False
    :return: A list of paths to the tiff files within the osm region.
    :rtype: list[os.PathLike]
    """
    logging.info("Getting tiffs around point")

    file_list = u4files.get_file_list_tiff(source_file_path)
    points = u4spatial.select_points_point(point, radius, file_list)
    out_file_list = np.array(file_list)[points.source_ind]
    return np.unique(out_file_list).tolist()


def ndarray_to_geotiff(
    Z: np.ndarray,
    bounds: tuple,
    file_path: os.PathLike,
    crs: str,
    compress: str = "none",
):
    """Saves a numpy nd array and its bounds to a georeferenced tiff.

    :param Z: The array
    :type Z: np.ndarray
    :param bounds: The edges of the array in real world coordinates, (min x, max x, min y, max y)
    :type bounds: tuple
    :param file_path: The path to save the tiff to.
    :type file_path: os.PathLike
    :param crs: The coordinate system to use.
    :type crs: str
    :param compress: The compression for the tiff file, defaults to "none".
    :type compress: str
    """
    height, width = Z.shape
    resx = (bounds[1] - bounds[0]) / width
    resy = (bounds[3] - bounds[2]) / height
    transform = rio.Affine.translation(
        bounds[0] - resx, bounds[2]
    ) * rio.Affine.scale(resx, resy)

    with rio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=Z.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
        compress=compress,
    ) as dst:
        dst.write(Z, 1)


def get_clipped_tiff_list(
    tiff_file_list: list[os.PathLike],
    gpkg_path: os.PathLike,
    overwrite: bool = False,
    use_parallel: bool = True,
) -> list[os.PathLike]:
    """Gets all tiffs clipped by the shapes in gpkg. Uses SQL logic to
    determine which shapes to take. The buffer distances are taken from `u4py.
    utils.config`. Each tiff file is clipped individually rather than creating a large single tiff. This is outsourced to several parallel workers.

    :param tiff_file_list: The file list of tiffs to clip.
    :type tiff_file_list: list[os.PathLike]
    :param gpkg_path: The path to the gpkg file containing all features for clipping.
    :type gpkg_path: os.PathLike
    :param overwrite: Whether to overwrite the output tiffs, defaults to False
    :type overwrite: bool, optional
    :param use_parallel: Whether to use parallel processing, defaults to True
    :type use_parallel: bool, optional
    :return: A list of clipped tiffs.
    :rtype: list[os.PathLike]
    """
    logging.info("Getting clipped tiffs using GPKG shapes and SQL.")
    base_path = u4files.multi_split(tiff_file_list[0], 2)
    ctiff_fol = os.path.join(base_path, f"clipped_tiffs_gpkg")
    os.makedirs(ctiff_fol, exist_ok=True)

    clipped_tiff_list = []
    if os.path.exists(ctiff_fol) and not overwrite:
        logging.info("Loading existing data")
        clipped_fnames = [
            tf for tf in os.listdir(ctiff_fol) if tf.endswith(".tif")
        ]
        clipped_tiff_list = [
            os.path.join(ctiff_fol, tf) for tf in clipped_fnames
        ]
        tiff_file_list = [
            tfp
            for tfp in tiff_file_list
            if os.path.split(tfp)[1] not in clipped_fnames
        ]
        if len(tiff_file_list) > 0:
            clipped_tiff_list = []

    if not clipped_tiff_list:
        logging.info("Clipping tiff files.")
        args = [(tfp, gpkg_path, ctiff_fol) for tfp in tiff_file_list]

        if use_parallel:
            u4proc.batch_mapping(args, batch_clip_tiff, desc="Masking Rasters")
        else:
            for arg in tqdm(
                args,
                total=len(tiff_file_list),
                desc="Masking Rasters",
                leave=False,
            ):
                batch_clip_tiff(arg)

        clipped_tiff_list = [
            os.path.join(ctiff_fol, tf)
            for tf in os.listdir(ctiff_fol)
            if tf.endswith(".tif")
        ]
    return clipped_tiff_list


def batch_get_thresholded_contours(args):
    gdf = get_thresholded_contours(*args, save_intermediate=False)
    return gdf


def get_thresholded_contours(
    tiff_file_path: os.PathLike,
    levels: list,
    threshold: float,
    overwrite: bool = False,
    save_intermediate: bool = True,
) -> gp.GeoDataFrame:
    """Gets contours around regions of specific ground motions (`levels`) that
    are larger than the `threshold` area (in m²). Data is stored as a
    shapefile for faster access

    :param tiff_file_path: The input tiff file to create contours.
    :type tiff_file_path: os.PathLike
    :param levels: The levels of the contours
    :type levels: list
    :param threshold: The minimum area for detection
    :type threshold: float
    :param overwrite: Whether to overwrite the existing results, defaults to False
    :type overwrite: bool, optional
    :return: A GeoDataFrame containing polygon shapes including some more info.
    :rtype: gp.GeoDataFrame
    """
    logging.debug("Getting thresholded contours")

    logging.debug("Setting up paths")
    base_path, fname = os.path.split(tiff_file_path)
    out_folder = os.path.join(base_path, "thresholded tiffs")
    os.makedirs(out_folder, exist_ok=True)
    fname = os.path.splitext(fname)[0]
    out_fname = fname.replace("clipped_tiffs", "thresh_contours")
    out_path = os.path.join(out_folder, out_fname + ".shp")

    if os.path.exists(out_path) and not overwrite:
        logging.debug("Loading from file")
        gdf = gp.read_file(out_path)
    else:
        logging.debug("Generating new results")
        coords, zz, crs = extract_xyz_tiff(tiff_file_path)
        gdf = u4spatial.contour_shapes(coords, zz, levels, threshold, crs)
        if len(gdf) > 0:
            if save_intermediate:
                gdf.to_file(out_path)
    return gdf


def wkt_from_tiff_bounds(fpath: os.PathLike, out_crs: str = "") -> str:
    """Returns the boundaries of a tiff file as a wkt string.

    :param fpath: The path to the tiff file.
    :type fpath: os.PathLike
    :param out_crs: The output CRS, defaults to "" which keeps the tiff's CRS
    :type out_crs: str, optional
    :return: A wkt string of the bounding polygon.
    :rtype: str
    """

    with rio.open(fpath) as tile:
        wkt = u4spatial.bounds_to_polygon(tile.bounds).wkt
        if out_crs:
            wkt = (
                gp.GeoDataFrame(geometry=[shapely.from_wkt(wkt)], crs=tile.crs)
                .to_crs(out_crs)
                .to_wkt()
                .geometry[0]
            )
    tile.close()

    return wkt


def calculate_volume_in_shape(
    shapes: gp.GeoDataFrame, tiff_folder: os.PathLike
) -> list:
    """Calculates the total volume moved for each shape in `shapes` using the data from the DEM found in `tiff_folder`. Also computes some basic statistics for each area.

    :param shapes: The shapes in a geodataframe.
    :type shapes: gp.GeoDataFrame
    :param tiff_folder: The folder where to find the DEM data (in `*.tiff` format.)
    :type tiff_folder: os.PathLike
    :return: A list of the calculated volume for each shape in shapes.
    :rtype: gp.GeoDataFrame
    """

    coverage = get_tiff_coverage(tiff_folder)
    if shapes.crs != coverage.crs:
        coverage = coverage.to_crs(shapes.crs)
    volumes = []
    volumes_removed = []
    volumes_added = []
    volumes_moved = []
    for geom in shapes.geometry:
        part = coverage.clip(geom)
        if len(part) > 0:
            vol = 0
            vol_removed = 0
            vol_added = 0
            vol_moved = 0
            flist = part.path.to_list()
            for fp in flist:
                v = u4spatial.compute_for_raster_in_geom(
                    fp, geom, np.nansum, shapes.crs
                )
                if not v is None:
                    vol += v
                v = u4spatial.compute_for_raster_in_geom(
                    fp, geom, u4spatial.vol_removed, shapes.crs
                )
                if not v is None:
                    vol_removed += v
                v = u4spatial.compute_for_raster_in_geom(
                    fp, geom, u4spatial.vol_added, shapes.crs
                )
                if not v is None:
                    vol_added += v
                v = u4spatial.compute_for_raster_in_geom(
                    fp, geom, u4spatial.vol_moved, shapes.crs
                )
                if not v is None:
                    vol_moved += v
        else:
            vol = np.nan
            vol_removed = np.nan
            vol_added = np.nan
            vol_moved = np.nan
        volumes.append(vol)
        volumes_removed.append(vol_removed)
        volumes_added.append(vol_added)
        volumes_moved.append(vol_moved)
    return {
        "volumes": volumes,
        "volumes_removed": volumes_removed,
        "volumes_added": volumes_added,
        "volumes_moved": volumes_moved,
    }


def get_tiff_coverage(
    tiff_folder: os.PathLike, overwrite: bool = False
) -> gp.GeoDataFrame:
    """Reads the coverage from all tiff files in the tiff folder and stores them as polygons for easier intersection of geometries.

    :param tiff_folder: The folder where the tiff files are located.
    :type tiff_folder: os.PathLike
    :param overwrite: Whether to overwrite the output shapefile, defaults to False
    :type overwrite: bool, optional
    :return: The geodataframe with the rectangles, paths to the file are stored as separate column `path`.
    :rtype: gp.GeoDataFrame
    """
    coverage_path = os.path.join(tiff_folder, "tiff_coverage.shp")

    if not os.path.exists(coverage_path) or overwrite:
        file_list = u4files.get_file_list_tiff(tiff_folder)
        geometry = []
        for fp in file_list:
            with rio.open(fp) as raster:
                geometry.append(u4spatial.bounds_to_polygon(raster.bounds))
                crs = raster.crs

        coverage = gp.GeoDataFrame(
            data={"geometry": geometry, "path": file_list}, crs=crs
        )
        coverage.to_file(coverage_path)
    else:
        coverage = gp.read_file(coverage_path)

    return coverage


def get_terrain(
    file_path: os.PathLike,
    terrain_feature: str = "slope",
    overwrite: bool = False,
) -> os.PathLike:
    """
    Uses GDAL to calculate the specified terrain feature of a geotiff file,
    e.g. the `slope`, `aspect`, `hillshade`, `multi_hillshade`. Stores the
    results in a separate folder in the input directory for faster access.

    :param file_path: The path to the tiff file.
    :type file_path: os.PathLike
    :param terrain_feature: The terrain feature, defaults to "slope"
    :type terrain_feature: str, optional
    :param overwrite: Whether to overwrite the existing slope files, defaults to False
    :type overwrite: bool, optional
    :return: The file_path to the file where the terrain data is stored.
    :rtype: os.PathLike
    """

    processing_keywords = {
        "slope": {"slopeFormat": "percent"},
        "aspect": {"zeroForFlat": True},
        "hillshade": {"zFactor": 3},
        "multi_hillshade": {"zFactor": 3, "multiDirectional": True},
    }
    processing = {
        "slope": "slope",
        "aspect": "aspect",
        "hillshade": "hillshade",
        "multi_hillshade": "hillshade",
    }

    source_path, file_name = os.path.split(file_path)
    base_path = os.path.split(source_path)[0]
    out_folder = os.path.join(base_path, terrain_feature)
    os.makedirs(out_folder, exist_ok=True)
    if not file_name.endswith(".tiff"):
        out_path = os.path.join(out_folder, file_name + ".tiff")
    else:
        out_path = os.path.join(out_folder, file_name)

    if not os.path.exists(out_path) or overwrite:
        logging.debug(f"Creating slope for {file_name}.")
        gdal_opts = gdal.DEMProcessingOptions(
            **processing_keywords[terrain_feature]
        )  # For now the Options seem not to work properly, DEMProcessing does
        # not accept further kwargs, even though its mentioned in the
        # documentation.
        gdal.DEMProcessing(out_path, file_path, processing[terrain_feature])
    else:
        logging.debug(f"Slope file found at {out_path}.")

    return out_path


def calculate_terrain_in_shapes(
    shapes: gp.GeoDataFrame, tiff_folder: os.PathLike, terrain_feature: str
) -> dict:
    """
    Calculates the specified terrain feature for each shape in `shapes` using
    the data from the DEM found in `tiff_folder`. Also computes some basic
    statistics for each area.

    :param shapes: The shapes in a geodataframe.
    :type shapes: gp.GeoDataFrame
    :param tiff_folder: The folder where to find the DEM data (in `*.adf` format.)
    :type tiff_folder: os.PathLike
    :param terrain_feature: One of `'slope'`, `'aspect'`, `'hillshade'`, `'multi_hillshade'`
    :type terrain_feature: str
    :return: The calculated values as dictionary with mean, median and stddev.
    :rtype:dict
    """
    file_list = u4files.get_file_list_adf(tiff_folder)
    points = u4spatial.select_points_region(shapes, file_list)
    file_list = [file_list[ii] for ii in points.source_ind]
    terrain_data = {
        "mean": [],
        "median": [],
        "std": [],
    }

    if len(file_list) > 0 and len(shapes) > 0:
        for shape in shapes.geometry.to_list():
            # Loop over all tiff files because a shape may overlap boundaries
            data = []
            for fp in file_list:
                fp_terrain = get_terrain(
                    fp, terrain_feature=terrain_feature, overwrite=False
                )
                terrain = u4spatial.compute_for_raster_in_geom(
                    fp_terrain, shape, u4spatial.get_values, shapes.crs
                )
                if terrain is not None:
                    data.extend(terrain)
            if len(data) > 0:
                data = np.array(data)
                if terrain_feature == "aspect":
                    terrain_data["mean"].append(
                        np.rad2deg(spstats.circmean(np.deg2rad(data)))
                    )
                    terrain_data["std"].append(
                        np.rad2deg(spstats.circstd(np.deg2rad(data)))
                    )

                else:
                    terrain_data["mean"].append(np.mean(data))
                    terrain_data["median"].append(np.median(data))
                    terrain_data["std"].append(np.std(data))
    return terrain_data


def get_tiff_regions(
    region: gp.GeoDataFrame, folder: os.PathLike
) -> list[os.PathLike]:
    """Gets a list of all tiff files within the region

    :param region: The region to search for tiff files.
    :type region: gp.GeoDataFrame
    :param folder: The folder where the raster data is stored
    :type folder: os.PathLike
    :return: A list of all tiffs that intersect the region.
    :rtype: list[os.PathLike]
    """
    coverage = get_tiff_coverage(folder)
    if region.crs != coverage.crs:
        coverage = coverage.to_crs(region)
    part = coverage.clip(region)
    if len(part) > 0:
        return part.path.to_list()
    else:
        logging.info("No raster data in specified region found.")
        return []
