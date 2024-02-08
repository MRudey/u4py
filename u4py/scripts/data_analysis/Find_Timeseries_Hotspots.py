"""
Uses the geotiffs for maximum sinusoidal and linear activity and detects
regions with larger than average movement.

This script uses 2d histogram binning with hexagons (`matplotlib.hexbin`).
"""

import os
from pathlib import Path

import u4py.analysis.spatial as u4spatial
import u4py.plotting.plots as u4plots
import u4py.utils.projects as u4proj


def main():
    # Setting Paths
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Find_Timeseries_Hotspots.u4project"
        ).expanduser(),
        required=["base_path", "output_path", "places_path"],
        interactive=False,
    )
    linear_path = os.path.join(
        project["paths"]["output_path"],
        "inversion_results_all_2023_tiffs",
        "linear_trend_50_inversion_results_all_2023.tif",
    )
    seasonal_path = os.path.join(
        project["paths"]["output_path"],
        "inversion_results_all_2023_tiffs",
        "max_vals_annual_50_inversion_results_all_2023.tif",
    )

    # Linear Hotspots
    lin_cen, lin_val, lin_crs = u4plots.plot_hotspots(
        linear_path,
        thresh=(-2.5, 2.5),
        nbins=10,
        min_count=5,
        output_filepath=os.path.join(
            project["paths"]["output_path"], "hotspots_linear.pdf"
        ),
        title="Linear Hotspots from PSI",
    )
    lin_gdf = u4spatial.xy_data_to_gdf(
        lin_cen[:, 0], lin_cen[:, 1], lin_val, crs=lin_crs
    )
    lin_gdf.to_file(
        os.path.join(project["paths"]["places_path"], "hotspots_linear.shp")
    )

    # Seasonal Hotspots
    seas_cen, seas_val, seas_crs = u4plots.plot_hotspots(
        seasonal_path,
        thresh=2.5,
        nbins=10,
        min_count=5,
        output_filepath=os.path.join(
            project["paths"]["output_path"], "hotspots_seasonal.pdf"
        ),
        title="Seasonal Hotspots from PSI",
    )
    seas_gdf = u4spatial.xy_data_to_gdf(
        seas_cen[:, 0], seas_cen[:, 1], seas_val, crs=seas_crs
    )
    seas_gdf.to_file(
        os.path.join(project["paths"]["places_path"], "hotspots_seasonal.shp")
    )


if __name__ == "__main__":
    main()
