import os
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt

import u4py.addons.gas_storage as u4gas
import u4py.io.files as u4files
import u4py.io.gpkg as u4gpkg
import u4py.plotting.axes as u4ax
import u4py.utils.convert as u4convert
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Examples\Bericht_FFM_BBD_EGMS.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "piloten_path",
            "output_path",
            "processing_path",
        ],
        interactive=False,
    )

    # Setting up Paths
    bbd_fpath = os.path.join(
        project["paths"]["base_path"], "Data_2023", "hessen_l3_clipped.gpkg"
    )
    egms_fpath_vert1 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2015-2021",
        "EGMS_2015_2021_vertikal.gpkg",
    )
    egms_fpath_vert2 = os.path.join(
        project["paths"]["base_path"],
        "Data_EGMS_2018-2022",
        "EGMS_2018_2022_vertikal.gpkg",
    )
    regions = u4files.get_rois(project["paths"]["piloten_path"])
    crumstadt_roi = gp.GeoDataFrame(geometry=[regions[4][1]], crs="EPSG:32632")

    # Load bbd dataset
    bbd_v_path = os.path.join(
        project["paths"]["processing_path"], "bbd_v_crumstadt.gpkg"
    )
    if not os.path.exists(bbd_v_path):
        bbd_v_gdf = u4gpkg.load_gpkg_data_region_ogr(
            crumstadt_roi,
            bbd_fpath,
            "vertikal",
        )
        bbd_v_gdf.to_file(bbd_v_path)
    else:
        bbd_v_gdf = gp.read_file(bbd_v_path)
    bbd_v = u4convert.reformat_gdf_to_dict(bbd_v_gdf)

    # Load egms1 dataset
    egms1_v_path = os.path.join(
        project["paths"]["processing_path"], "egms1_v_crumstadt.gpkg"
    )
    if not os.path.exists(egms1_v_path):
        egms1_v_gdf = u4gpkg.load_gpkg_data_region_ogr(
            crumstadt_roi,
            egms_fpath_vert1,
            "EGMS_2015_2021_vertikal",
        )
        egms1_v_gdf.to_file(egms1_v_path)
    else:
        egms1_v_gdf = gp.read_file(egms1_v_path)
    egms1_v = u4convert.reformat_gdf_to_dict(egms1_v_gdf)

    # Load egms2 dataset
    egms2_v_path = os.path.join(
        project["paths"]["processing_path"], "egms2_v_crumstadt.gpkg"
    )
    if not os.path.exists(egms2_v_path):
        egms2_v_gdf = u4gpkg.load_gpkg_data_region_ogr(
            crumstadt_roi,
            egms_fpath_vert2,
            "EGMS_2018_2022_vertikal",
        )
        egms2_v_gdf.to_file(egms2_v_path)
    else:
        egms2_v_gdf = gp.read_file(egms2_v_path)
    egms2_v = u4convert.reformat_gdf_to_dict(egms2_v_gdf)
    date_gas, _, level_gas = u4gas.load_gas_data(
        os.path.join(
            project["paths"]["ext_path"], "Inventory Turnover Data_23.txt"
        )
    )

    # Plotting
    fig, axes = plt.subplots(nrows=4, sharex=True)

    u4ax.plot_timeseries(
        bbd_v["time"], bbd_v["timeseries"], ax=axes[0], color="C0"
    )
    u4ax.plot_timeseries(
        egms1_v["time"], egms1_v["timeseries"], ax=axes[1], color="C1"
    )
    u4ax.plot_timeseries(
        egms2_v["time"], egms2_v["timeseries"], ax=axes[2], color="C2"
    )
    axes[3].plot(date_gas, level_gas, color="k")

    axes[1].set_ylabel("Bodenbewegung (mm)")
    axes[3].set_ylabel("Füllstand (%)")
    axes[-1].set_xlabel("Zeitraum")
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])
    axes[0].set_ylim(-10, 10)

    for ii, lbl in enumerate(
        ["BBD", "EGMS 2015-2021", "EGMS 2018-2022", "Gasspeicher"]
    ):
        axes[ii].annotate(lbl, (0.01, 0.05), xycoords="axes fraction")

    fig.tight_layout()
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "PSI_vs_Gas.png")
    )
    fig.savefig(
        os.path.join(project["paths"]["output_path"], "PSI_vs_Gas.pdf")
    )


if __name__ == "__main__":
    main()
