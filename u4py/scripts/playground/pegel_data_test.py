"""
Extracts and visualizes river coordinates close to a station along a certain river.
"""

import contextily
import matplotlib.pyplot as plt

import u4py.addons.rivers as u4rivers
import u4py.analysis.spatial as u4spatial


def main():
    file_path = (
        r"~\Documents\ArcGIS\ExternalData\Wasserstand\23960709Day.MeanW.zrx"
    )
    data = u4rivers.get_pegel_data(file_path)

    station = data["23980353"]
    river = u4spatial.river_coordinates_from_osm(
        station["name"], station["water"]
    )
    if not river.empty:
        fig, ax = plt.subplots()
        # ax.plot(
        #     data["23960709"]["level"]["time"],
        #     data["23960709"]["level"]["level"],
        #     ".",
        # )
        # ax.plot(
        #     data["23960709"]["temperature"]["time"],
        #     data["23960709"]["temperature"]["temperature"],
        #     ".",
        # )
        river.plot(ax=ax)
        contextily.add_basemap(
            ax,
            crs=river.crs.to_string(),
            # zoom=15,
            source=contextily.providers.OpenStreetMap.Mapnik,
        )
        plt.show()


if __name__ == "__main__":
    main()
