import os

import contextily as cx
import geopandas
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as rioplot
import u4py.analysis.spatial as u4spatial


def main():
    bld_path = (
        r"C:\Users\Michael Rudolf\Documents\ArcGIS\Places\vg2500_bld.dbf"
    )
    folder_path, _ = os.path.split(bld_path)
    temp_map_path = os.path.join(folder_path, "temp_map.tif")
    hessen_map_path = os.path.join(folder_path, "hessen_map.tif")
    bld = geopandas.read_file(bld_path)
    hessen = bld[bld["GEN"] == "Hessen"]
    fig, ax = plt.subplots()
    hessen.plot(ax=ax)
    (w, s, e, n) = hessen.total_bounds
    _ = cx.bounds2raster(w, s, e, n, temp_map_path, ll=True, zoom=10)
    u4spatial.reproject_raster(temp_map_path, hessen_map_path, "EPSG:32632")
    with rasterio.open(hessen_map_path) as hessen_map:
        rioplot.show(hessen_map)
    # cx.add_basemap(ax=ax, crs=hessen.crs.to_string(), zorder=0)


if __name__ == "__main__":
    main()
