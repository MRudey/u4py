"""
Contains helper functions to prepare and rearrange data for working with
ADATools, such as automatic creation of readmaps and extracting PS data from
GPKGs and saving them to shapefiles.
"""

import logging
import os
from collections import OrderedDict
from multiprocessing import Pool
from typing import Iterable, Tuple

import geopandas as gp
import shapely as shp
from tqdm import tqdm

import u4py.io.files as u4files
import u4py.io.sql as u4sql
import u4py.utils.config as u4config


def create_adafinder_read_map(
    file_path: os.PathLike,
    position_x: int = 0,
    position_y: int = 1,
    position_velocity: int = 2,
    position_time_series: int = 3,
    n_values_time_series: int = 0,
    have_lambda_fi: int = 0,
    position_lambda: int = 0,
    position_fi: int = 0,
    output_ts_field_names: list = [],
):
    """Creates a readmap for ADAfinder to read the shapefiles. Use
    `dump_dbf_header` from ADAtools to see the indices for each column.

    :param file_path: The path where to store the readmap.
    :type file_path: os.PathLike
    :param position_x: The column index of the x coordinate, defaults to 0
    :type position_x: int, optional
    :param position_y: The column index of the y coordinate, defaults to 1
    :type position_y: int, optional
    :param position_velocity: The column index of the velocity field, defaults to 2
    :type position_velocity: int, optional
    :param position_time_series: The starting column index of the time series, defaults to 3
    :type position_time_series: int, optional
    :param n_values_time_series: The number of values in the timeseries, defaults to 0 = automatically determined from the length of `output_ts_field_names`
    :type n_values_time_series: int, optional
    :param have_lambda_fi: 1 if the data has a set of lambda/fi coordinates, defaults to 0
    :type have_lambda_fi: int, optional
    :param position_lambda: The column index of the lambda coordinate, defaults to 0
    :type position_lambda: int, optional
    :param position_fi: The column index of the fi coordinate, defaults to 0
    :type position_fi: int, optional
    :param output_ts_field_names: The list of field names for the timeseries, ideally formatted as "D%Y%m%d", defaults to []
    :type output_ts_field_names: list, optional
    """
    if not n_values_time_series and output_ts_field_names:
        n_values_time_series = len(output_ts_field_names)
    elif n_values_time_series and not output_ts_field_names:
        output_ts_field_names = [
            f"D{n:06}" for n in range(n_values_time_series)
        ]
    else:
        ValueError("Timeseries Values or Output Field Names not correct.")

    with open(file_path, "wt") as op_file:
        op_file.write(f"POSITION_X = {position_x}\n")
        op_file.write(f"POSITION_Y = {position_y}\n")
        op_file.write(f"POSITION_VELOCITY = {position_velocity}\n")
        op_file.write(f"POSITION_TIME_SERIES = {position_time_series}\n")
        op_file.write(f"N_VALUES_TIME_SERIES = {n_values_time_series}\n")
        op_file.write(f"HAVE_LAMBDA_FI = {have_lambda_fi}\n")
        op_file.write(f"POSITION_LAMBDA = {position_lambda}\n")
        op_file.write(f"POSITION_FI = {position_fi}\n")

        ts_field_str = ""
        for ii, ts in enumerate(output_ts_field_names):
            if ii == 0 or ii % 5:
                ts_field_str += " " + ts
            else:
                ts_field_str += " \\\n                        " + ts
        op_file.write(f"OUTPUT_TS_FIELD_NAMES ={ts_field_str}")


def load_region_as_gdf(
    gpkg_path: os.PathLike, table: str, region: gp.GeoDataFrame = []
) -> Tuple[gp.GeoDataFrame, list]:
    """Loads data within the given region from `fname` and the `table`.
    Returns a geodatabase that has suitable format for ADAtools shapefiles and
    a list of timestamps for generation of an adequate readmap.

    If the region is empty, extracts the whole table.

    :param gpkg_path: The filename of the gpkg file.
    :type gpkg_path: str
    :param table: The sql table name to get the data from.
    :type table: str
    :param region: The region to extract the data in, defaults to [].
    :type region: gp.GeoDataFrame
    :return: The extracted points and list with timestamp strings.
    :rtype: Tuple[gp.GeoDataFrame, list]
    """

    if region:
        data = u4sql.load_gpkg_data_region(
            region,
            gpkg_path,
            table=table,
        )
    else:
        data = u4sql.table_to_dict(gpkg_path, table)

    if data:
        logging.info("Reformatting for new structure")
        time_stamps = [t.strftime("D%Y%m%d") for t in data["time"]]
        velocity = data["timeseries"][:, -1] / (
            (data["time"][-1] - data["time"][0]).days / 365.25
        )
        data_gdf = {
            "geometry": [
                shp.Point(x, y) for x, y in zip(data["x"], data["y"])
            ],
            "X": data["x"],
            "Y": data["y"],
            "Velocity": velocity,
        }
        for ii, t_key in enumerate(time_stamps):
            data_gdf[t_key] = data["timeseries"][:, ii]
        gdf = gp.GeoDataFrame(data=data_gdf, crs="EPSG:32632")
        return gdf, time_stamps
    else:
        return [], []


def convert_gpkg_to_shp(fname: str, project: dict):
    """Converts the data from the gpkg to a shape file and readmap for table.

    :param fname: The filename of the gpkg file.
    :type fname: str
    :param project: The project config
    :type project: dict

    Subdivides the tables into smaller shape files depending on the number of
    entries in the table. The resulting subdivisions are stored in shapefiles
    for easier later access and quality control.
    """
    fpath = os.path.join(project["paths"]["psi_path"], fname)

    args = []
    ii = 0
    # For ADAtools the maximum size for the dBASE is 2 GB. This means a
    # shapefile can have about 70M entries for points, we have about 360
    # fields per entry, so 150k is a good estimate for maximum size.

    subregs = u4sql.get_subdivided_regions(fpath, "vertikal", 150000)
    subregs_gdf = gp.GeoDataFrame(geometry=subregs, crs="EPSG:32632")
    subregs_gdf.to_file(
        os.path.join(project["paths"]["psi_path"], "subregions.shp")
    )
    for jj, reg in enumerate(subregs):
        ii += 1
        args.append((fpath, "vertikal", reg, project, ii, jj + 1))
        ii += 1
        args.append((fpath, "Ost_West", reg, project, ii, jj + 1))

    # for arg in args:
    #     conversion_worker(arg)
    ncpu = min(len(args), u4config.cpu_count)
    with Pool(ncpu) as p:
        logging.info("Starting Parallel Conversion")
        list(
            tqdm(
                p.imap_unordered(conversion_worker, args),
                total=len(args),
                desc="Processing Layers",
                leave=False,
                position=0,
            )
        )


def conversion_worker(args: Iterable):
    """Parallel processing wrapper for file conversion.

    :param args: A set of arguments for processing.
    :type args: Iterable
    """
    gpkg_path, table, region, project, ii, subreg = args
    if "Geschwindigkeit" in table:
        pass
    else:
        gdf, time_stamps = load_region_as_gdf(gpkg_path, table, region)
        logging.info("Writing input_points to Shapefile")
        tblshrt = table.replace("Zeitreihe_", "")
        if subreg:
            tblshrt += f"_{subreg}"

        # Use custom saving function with progress display
        u4files.to_file_fiona(
            gdf,
            os.path.join(
                project["paths"]["output_path"],
                f"input_points_{tblshrt}.shp",
            ),
            driver="ESRI Shapefile",
            position=ii,
        )

        logging.info("Creating readmap for input_points")
        create_adafinder_read_map(
            os.path.join(
                project["paths"]["output_path"],
                f"input_points_{tblshrt}_readmap.op",
            ),
            position_x=0,
            position_y=1,
            position_velocity=2,
            position_time_series=3,
            output_ts_field_names=time_stamps,
        )


def create_adafinder_config(
    in_path: os.PathLike, project: dict, **kwargs
) -> os.PathLike:
    """Creates a config for use with ADAfinder CLI

    :param in_path: The path to the input points shapefile or csvfile.
    :type in_path: os.PathLike
    :param project: The u4py project config
    :type project: dict
    :param **kwargs: Additional arguments passed to ADAfinder config, with keys in UPPERCASE.
    :type **kwargs: dict
    :return: The path to the ADAfinder config
    :rtype: os.PathLike
    """
    logging.info("Creating config")
    file_format = {".shp": "SHAPEFILE", ".csv": "CSV-COMMA"}

    in_folder, in_file = os.path.split(in_path)
    in_fname, in_format = os.path.splitext(in_file)
    adacfg_path = os.path.join(in_folder, f"{in_fname}_adafinder.op")

    out_folder = project["paths"]["output_path"]
    os.makedirs(out_folder, exist_ok=True)
    output_shapefile_adas = os.path.join(out_folder, f"{in_fname}_adas.shp")
    output_points = os.path.join(out_folder, f"{in_fname}_points.csv")
    output_shapefile_buffered_adas = os.path.join(
        out_folder, f"{in_fname}_buffered_adas.shp"
    )
    output_buffered_points = os.path.join(
        out_folder, f"{in_fname}_buffered_points.csv"
    )

    adacfg = OrderedDict(
        INPUT_POINTS=in_path,
        INPUT_POINTS_FORMAT=file_format[in_format],
        INPUT_POINTS_READMAP=in_path.replace(".shp", "_readmap.op"),
        INPUT_SHAPEFILE_BOUNDARIES="-",
        OUTPUT_SHAPEFILE_ADAS=output_shapefile_adas,
        OUTPUT_POINTS=output_points,
        OUTPUT_POINTS_FORMAT="CSV-COMMA",
        POINT_SUBSET_TO_WRITE=2,
        COMPUTE_BUFFERED_ADAS=False,
        OUTPUT_SHAPEFILE_BUFFERED_ADAS=output_shapefile_buffered_adas,
        OUTPUT_BUFFERED_POINTS=output_buffered_points,
        OUTPUT_BUFFERED_POINTS_FORMAT="CSV-COMMA",
        # Options -------------------------------------------------------------
        PURGE_ISOLATED_POINTS=False,
        PURGE_SMALL_CLUSTERS=False,
        ISOLATION_DISTANCE=100,  # 2*resolution
        ISOLATION_CLUSTER_SIZE=2,
        VELOCITY_THRESHOLD_MODE=3 * 0.65,  # Roughly matches 3*std
        VELOCITY_FACTOR_OR_THRESHOLD=1,
        ADA_RADIUS=65,  # 1.3*resolution
        ADA_MINIMUM_CLUSTER_SIZE=5,
        ADA_BUFFER_SIZE=50,
        # Advanced options ----------------------------------------------------
        ADA_DISPLAY_MODE=1,
        ALLOW_OVERLAPPING_ADAS=False,
        ADA_SHIFT_TO_CM=2,
        ADA_CIRCLE_STEPS=40,
        N_VALUES_MEAN_DEFORMATION=6,
        TNITABLE_N_VALUES=3,
        TNITABLE_THRESHOLDS=[0.53, 0.7, 0.84],
        SNITABLE_N_VALUES=3,
        SNITABLE_THRESHOLDS=[0.53, 0.7, 0.84],
        QITABLE_CLASSES=[1, 1, 2, 4, 1, 2, 3, 4, 2, 3, 3, 4, 4, 4, 4, 4],
    )

    adacfg.update(**kwargs)

    with open(adacfg_path, "wt") as cfgf:
        for k, v in adacfg.items():
            if isinstance(v, str):
                cfgf.write(f"{k} = {v}\n")
            elif isinstance(v, bool):
                if v:
                    cfgf.write(f"{k} = YES\n")
                else:
                    cfgf.write(f"{k} = NO\n")
            elif isinstance(v, Iterable):
                cfgf.write(f"{k} = {' '.join([str(va) for va in v])}\n")
            elif isinstance(v, int):
                cfgf.write(f"{k} = {v}\n")
            elif isinstance(v, float):
                cfgf.write(f"{k} = {v:.2f}\n")
    return adacfg_path


def merge_shps(project: dict, ending: str, direction: str):
    """Merges output of several ADAfinder runs into a single shapefile.

    :param project: The project config
    :type project: dict
    :param ending: The ending to identify the type of results.
    :type ending: str
    :param direction: The direction (vertikal, Ost_West) of the results.
    :type direction: str
    """
    shp_file_list = [
        os.path.join(project["paths"]["output_path"], fp)
        for fp in os.listdir(project["paths"]["output_path"])
        if fp.endswith(f"{ending}") and direction in fp
    ]
    outfilep = os.path.join(
        project["paths"]["output_path"],
        "merged_ada_results",
        f"merged_{direction}_{ending}",
    )
    if shp_file_list:
        data = dict()
        for shpf in tqdm(
            shp_file_list, desc="Reading files for merge", leave=False
        ):
            gdf = gp.read_file(shpf)
            if not data:
                for k in gdf.keys():
                    data[k] = []
            for k in gdf.keys():
                data[k].extend(gdf[k])
        fgdf = gp.GeoDataFrame(data=data, crs=gdf.crs)
        fgdf.to_file(outfilep)
