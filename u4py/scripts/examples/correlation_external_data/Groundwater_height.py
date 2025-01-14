import os
import pickle as pkl
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from tqdm import tqdm

import u4py.addons.groundwater as u4gw
import u4py.io.tiff as u4tiff
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"C:\Users\Michael Rudolf\Documents\ArcGIS\U4_projects\Groundwater_height.u4project"
        ).expanduser(),
        required=["base_path", "places_path", "ext_path", "diff_plan_path"],
        interactive=False,
    )
    wells_shp_path = r"C:\Users\Michael Rudolf\HESSENBOX-DA\Datenaustausch_Umwelt4.0\All_Correlations_GW_PSI.shp"
    wells_gdf = gp.read_file(wells_shp_path)
    wells_path = os.path.join(
        project["paths"]["ext_path"], "GWStände_2015", "GWStände_2015.pkl"
    )
    wells_data = u4gw.get_groundwater_data(wells_path)
    tiff_file_list = u4tiff.get_pointlist_tiff(
        wells_gdf, project["paths"]["diff_plan_path"]
    )
    with rio.open(tiff_file_list[0][1]) as dataset:
        raster_crs = dataset.crs
    wells_gdf = wells_gdf.to_crs(raster_crs)

    if (
        "z"
        not in wells_data[wells_gdf.iloc[tiff_file_list[0][0]]["name"]].keys()
    ):
        for ii, tfpath in tqdm(
            tiff_file_list,
            desc="Extracting well height from DEM1",
            leave=False,
        ):
            point = wells_gdf.iloc[ii]
            with rio.open(tfpath) as dataset:
                dem = dataset.read(1)
                height = dem[dataset.index(point.geometry.x, point.geometry.y)]
            wells_data[point["name"]]["z"] = height

        with open(wells_path, "wb") as well_file:
            pkl.dump(wells_data, well_file)

    avg_h = []
    idx = []
    zvals = []
    for ii, name in enumerate(wells_gdf.name.to_list()):
        z = wells_data[name]["z"]
        zvals.append(z)
        avg_h.append(z - np.median(wells_data[name]["height"]))
        idx.append(ii)

    wells_gdf = wells_gdf.assign(z=gp.pd.Series(zvals, idx))
    wells_gdf = wells_gdf.assign(avg_h=gp.pd.Series(avg_h, idx))
    wells_gdf.to_file(wells_shp_path)


if __name__ == "__main__":
    main()
