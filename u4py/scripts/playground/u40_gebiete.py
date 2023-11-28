import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import shapely

import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4pltfmt


def main():
    bld_path = Path(r"~\Documents\ArcGIS\Places\vg2500_bld.shp").expanduser()
    out_path = Path(r"~\Documents\ArcGIS\INSAR_plots").expanduser()
    bld_shp = gp.read_file(bld_path)
    ffm = (470500, 5548000, 479000, 5555000)

    ffm_shp = shapely.Polygon.from_bounds(*ffm)
    ffm_shp_c = shapely.Point(ffm_shp.centroid)
    ffm_gdf = gp.GeoSeries(data=ffm_shp, crs="EPSG:32632")
    ffm_gdf_c = gp.GeoSeries(data=ffm_shp_c, crs="EPSG:32632")

    crum = (462000, 5514000, 470000, 5520400)

    crum_shp = shapely.Polygon.from_bounds(*crum)
    crum_shp_c = shapely.Point(crum_shp.centroid)
    crum_gdf = gp.GeoSeries(data=crum_shp, crs="EPSG:32632")
    crum_gdf_c = gp.GeoSeries(data=crum_shp_c, crs="EPSG:32632")

    fig, ax = plt.subplots(dpi=150)
    figb = fig.bbox.bounds
    axb = ax.bbox.bounds
    width = 0.4
    left = (axb[2] + axb[0]) / figb[2] - width * 0.6
    bottom = 1 - (axb[3] + axb[1]) / figb[3]

    ffm_gdf.plot(
        ax=ax, facecolor="None", edgecolor="C0", zorder=4, linewidth=2
    )
    crum_gdf.plot(
        ax=ax, facecolor="None", edgecolor="C1", zorder=4, linewidth=2
    )

    in_ax = fig.add_axes([left, bottom, width, width])
    bld_shp.plot(ax=in_ax, facecolor="w", edgecolor="k", linewidth=0.5)
    bld_shp[bld_shp["GEN"] == "Hessen"].plot(
        ax=in_ax, facecolor="None", edgecolor="C3", linewidth=0.5
    )
    ffm_gdf_c.to_crs(bld_shp.crs).plot(ax=in_ax, edgecolor="C0", marker=".")
    crum_gdf_c.to_crs(bld_shp.crs).plot(ax=in_ax, edgecolor="C1", marker=".")
    in_ax.axis("off")
    ax.set_xlim(440000, 502500)
    u4ax.add_basemap(ax=ax, crs="EPSG:32632")
    u4pltfmt.map_style(ax=ax, crs="EPSG:32632")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "Overview_exJURSE.pdf"))
    fig.savefig(os.path.join(out_path, "Overview_exJURSE.png"))


if __name__ == "__main__":
    main()
