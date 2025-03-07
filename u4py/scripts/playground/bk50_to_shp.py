from pathlib import Path

import contextily
import geopandas as gp
import matplotlib.pyplot as plt
import shapely as shp

import u4py.addons.web_services as u4web
import u4py.analysis.spatial as u4spatial


def main():
    out_folder = Path(r"~\Documents\ArcGIS\Places\bodenkarte").expanduser()
    region = gp.GeoDataFrame(
        geometry=[
            u4spatial.bounds_to_polygon(
                (
                    8.486433384316346,
                    49.79822912154841,
                    8.577689428540465,
                    49.852937995056465,
                )
            )
        ],
        crs="EPSG:4326",
    ).to_crs("EPSG:32632")
    topsoil_data = u4web.query_hlnug(
        "boden/bfd50/MapServer",
        "BFD50_Bodenhauptgruppen",
        region=region,
        out_folder=out_folder,
        suffix=f"goddelau",
    ).clip(region)

    # fig, ax = plt.subplots()
    # region.plot(ax=ax, ec="k", fc="None")
    # contextily.add_basemap(ax, crs=region.crs)
    # plt.show()


if __name__ == "__main__":
    main()

    #  shp.Polygon(
    # 49.852937995056465,
    # , 8.577689428540465
