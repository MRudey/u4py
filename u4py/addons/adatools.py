"""
Contains helper functions to prepare and rearrange data for working with
ADATools, such as automatic creation of readmaps and extracting PS data from
GPKGs and saving them to shapefiles.
"""
import logging
import os
from typing import Tuple

import fiona
import geopandas as gp
import shapely as shp
from tqdm import tqdm

import u4py.analysis.spatial as u4spatial
import u4py.utils.files as u4files
import u4py.utils.sql as u4sql


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

    if len(region) > 0:
        data = u4spatial.load_gpkg_data_region(
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


def convert_gpkg_to_shp(
    fname: str, project: dict, tables: list = [], region: gp.GeoDataFrame = []
):
    """Converts the data from the gpkg to a shape file and readmap for table.

    - If the table is empty, extracts each table into a separate file.
    - If the region is empty, extracts the whole table.

    :param fname: The filename of the gpkg file.
    :type fname: str
    :param project: The project config
    :type project: dict
    :param tables: The sql table name to get the data from.
    :type tables: str
    :param region: The region to extract the data in, defaults to [].
    :type region: gp.GeoDataFrame
    """
    gpkg_path = os.path.join(project["paths"]["psi_path"], fname)
    if not tables:
        tables = fiona.listlayers(gpkg_path)
    for table in tqdm(tables, desc="Converting tables"):
        if "Geschwindigkeit" in table:
            pass
        else:
            gdf, time_stamps = load_region_as_gdf(gpkg_path, table, region)
            if len(gdf) > 0:
                logging.info("Writing input_points to Shapefile")
                tblshrt = table.replace("Zeitreihe_", "")
                # gdf.to_file(
                #     os.path.join(
                #         project["paths"]["output_path"],
                #         f"input_points_{tblshrt}.shp",
                #     )
                # )
                u4files.to_file_fiona(
                    gdf,
                    os.path.join(
                        project["paths"]["output_path"],
                        f"input_points_{tblshrt}.shp",
                    ),
                    driver="ESRI Shapefile",
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
            else:
                logging.info(f"{table} has no entries.")
