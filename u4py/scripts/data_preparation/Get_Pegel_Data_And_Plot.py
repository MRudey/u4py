"""
Extracts and visualizes river coordinates close to a station along a certain river.
"""

import os

import contextily
import matplotlib.pyplot as plt

import u4py.addons.rivers as u4rivers
import u4py.plotting.formatting as u4plotfmt
import u4py.utils.projects as u4proj


def main():
    # Setting Paths
    project = u4proj.get_project(
        required=["base_path", "ext_path", "places_path", "output_path"]
    )
    file_path = os.path.join(
        project["paths"]["ext_path"], "Wasserstand", "Pegel.pkl"
    )
    location_path = os.path.join(project["paths"]["places_path"], "Pegel.shp")

    # Load Data
    data = u4rivers.get_pegel_data(file_path)
    locations = u4rivers.get_pegel_locations(location_path)
    station_num = 23960709
    station = data[str(station_num)]

    # Create Figure
    fig = plt.figure(figsize=(7, 5), dpi=150)
    gs = fig.add_gridspec(nrows=2, height_ratios=(3, 1))

    # Map
    ax_map = fig.add_subplot(gs[0])
    locations[locations.PEGEL == station_num].plot(ax=ax_map)
    loc_geom = locations[locations.PEGEL == station_num].geometry
    x = loc_geom.x.iloc[0]
    y = loc_geom.y.iloc[0]
    ax_map.set_xlim(x - 1000, x + 1000)
    ax_map.set_ylim(y - 1000, y + 1000)
    contextily.add_basemap(
        ax_map,
        crs=locations.crs.to_string(),
        source=contextily.providers.OpenStreetMap.Mapnik,
    )
    u4plotfmt.map_style(ax=ax_map, divisor=500)
    # Data
    ax_data = fig.add_subplot(gs[1])
    ax_data.plot(station["level"]["time"], station["level"]["level"])
    ax_data.set_xlabel("Time")
    ax_data.set_ylabel("Water Level (cm)")
    fig.suptitle(f"{station['water']} in {station['name']}")

    # Save Plot
    out_folder = os.path.join(project["paths"]["output_path"], "Pegel_Data")
    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(
        os.path.join(out_folder, f"{station['water']}_{station['name']}.png")
    )
    fig.savefig(
        os.path.join(out_folder, f"{station['water']}_{station['name']}.pdf")
    )


if __name__ == "__main__":
    main()
