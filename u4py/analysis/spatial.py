import geopandas as gp
import h5py
import numpy as np
import osmnx
import rasterio as rio
import rasterio.warp as riowarp
import scipy.spatial as spspatial
import shapely.geometry as shpgeo


def get_features(in_dict: dict, features: list):
    """Gets specied features from input dictionary"""
    ind = []
    for feat in features:
        ind.extend(np.argwhere(in_dict["fclass"] == feat))
    coords = np.squeeze(in_dict["geometry"][ind])
    names = [
        str((n.encode("latin_1")).decode("utf8"))
        for n in np.squeeze(in_dict["name"][ind])
    ]

    return (names, coords)


def reproject_raster(in_path, out_path, output_crs):
    """reproject raster to project crs"""
    crs_out = {"init": output_crs}
    with rio.open(in_path) as src:
        transform, width, height = riowarp.calculate_default_transform(
            src.crs, crs_out, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()

        kwargs.update(
            {
                "crs": crs_out,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                riowarp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_out,
                    resampling=riowarp.Resampling.nearest,
                )
    return out_path


def get_cKDTree(file_path):
    """
    Loads all x and y coordinates from the given hdf5 file and returns a
    cKDTree for easy spatial lookup.
    """
    coords = get_coords(file_path)

    return spspatial.cKDTree(coords)


def get_coords(file_path):
    """
    Loads all x and y coordinates from the given hdf5 file
    """
    with h5py.File(file_path, "r") as h5file:
        coords = np.array(
            [(x, y) for x, y in zip(h5file["x"][()], h5file["y"][()])]
        )
    return coords


def select_points(query, psi_file_path):
    """
    Selects points from the specified file and crops them by the rectangles found in the given osm query
    """

    osm_data = osmnx.geometries_from_address(
        query["address"], tags=query["tags"]
    ).to_crs("EPSG:32632")

    coords = get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": shpgeo.MultiPoint(coords),
            "source_index": np.arange(len(coords)),
        },
        crs="EPSG:32632",
    )
    return points.clip(osm_data)


def select_points_region(region, psi_file_path):
    """
    Selects points from the specified file and crops them by the rectangles found in the given region
    """

    coords = get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": shpgeo.MultiPoint(coords),
            "source_index": np.arange(len(coords)),
        },
        crs="EPSG:32632",
    )
    return points.clip(region)


def select_points_point(point, radius, psi_file_path):
    region = gp.GeoDataFrame(
        {"geometry": [shpgeo.Point(point).buffer(radius)]}, crs="EPSG:32632"
    )
    coords = get_coords(psi_file_path)
    points = gp.GeoDataFrame(
        {
            "geometry": shpgeo.MultiPoint(coords),
            "source_index": np.arange(len(coords)),
        },
        crs="EPSG:32632",
    )
    return points.clip(region)
