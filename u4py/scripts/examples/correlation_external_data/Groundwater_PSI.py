import csv
import datetime
import os
from multiprocessing import Pool
from pathlib import Path

import geopandas as gp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import u4py.addons.groundwater as u4gw
import u4py.io.gpkg as u4gpkg
import u4py.plotting.axes as u4ax
import u4py.plotting.formatting as u4pltfmt
import u4py.utils.config as u4config
import u4py.utils.convert as u4conv
import u4py.utils.projects as u4proj


def main():
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\Examples\Groundwater_PSI.u4project"
        ).expanduser(),
        required=["psi_path", "ext_path", "processing_path", "output_path"],
        interactive=False,
    )
    use_parallel = False
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
    psi_paths = [
        (bbd_fpath, "vertikal"),
        (egms_fpath_vert1, "EGMS_2015_2021_vertikal"),
        (egms_fpath_vert2, "EGMS_2018_2022_vertikal"),
    ]
    processing_path = os.path.join(
        project["paths"]["processing_path"], "Groundwater_Well_Inversions"
    )
    os.makedirs(processing_path, exist_ok=True)
    out_path = os.path.join(project["paths"]["output_path"], "GW_PSI")
    os.makedirs(out_path, exist_ok=True)
    wells_gdf = gp.read_file(
        os.path.join(
            project["paths"]["ext_path"],
            "GWStände_2015",
            "GW_Stations.shp",
        )
    )
    wells_data = u4gw.get_groundwater_data(
        os.path.join(
            project["paths"]["ext_path"], "GWStände_2015", "GWStände_2015.pkl"
        )
    )

    args = [
        (wells_data, k, psi_paths, wells_gdf, out_path)
        for k in wells_data.keys()
    ]

    if use_parallel:
        with Pool(u4config.cpu_count - 2) as p:
            list(
                tqdm(
                    p.imap_unordered(parallel_make_plot, args),
                    total=len(args),
                    desc="Generating Plots",
                    leave=False,
                )
            )

    else:
        for arg in tqdm(args, desc="Generating Plots", leave=False):
            make_plot(*arg)


def parallel_make_plot(arg):
    make_plot(*arg)


def make_plot(
    wells_data: dict,
    k: str,
    psi_paths: list,
    wells_gdf: gp.GeoDataFrame,
    out_path: os.PathLike,
):
    data = wells_data[k]
    len_ts = len(data["time"])
    markers = ["s", "o", "."]
    data_name = ["BBD", "EGMS 2016-2021", "EGMS 2018-2022"]
    if len_ts > 100:
        fig, axes = plt.subplots(ncols=4, figsize=(20, 5), dpi=150)
        data_found = False
        wells_gdf[wells_gdf["name"] == k].plot(ax=axes[-1], color="k")
        wells_gdf[wells_gdf["name"] == k].buffer(500).plot(
            ax=axes[-1], fc="None", ec="k"
        )
        for ii, (fpath, table) in enumerate(psi_paths):
            well_psi, pts_gdf = load_data(wells_gdf, fpath, table, k)
            if well_psi:
                data_found = True
                try:
                    comp_data = u4gw.compare_gw_psi(well_psi, data)
                    axes[ii].plot(
                        comp_data["psi"],
                        comp_data["gw"],
                        color=f"C{ii}",
                        marker=markers[ii],
                    )
                    axes[ii].set_xlabel("Vertical displacement (mm)")
                    pts_gdf.to_crs(wells_gdf.crs).plot(
                        ax=axes[-1], color=f"C{ii}", marker=markers[ii]
                    )
                    csv_dir = os.path.join(out_path, f"{k}_comparison")
                    os.makedirs(csv_dir, exist_ok=True)
                    csv_fpath = os.path.join(
                        csv_dir, f"gw_psi_{data_name[ii]}.csv"
                    )
                    export_to_csv(comp_data, csv_fpath, well_psi)
                    pts_gdf.to_crs(wells_gdf.crs).to_file(
                        csv_fpath.replace(".csv", ".gpkg")
                    )

                except IndexError:
                    data_found = False
                    axes[ii].annotate(
                        "Not enough temporal overlap.",
                        (0.5, 0.5),
                        xycoords="axes fraction",
                        horizontalalignment="center",
                    )
            else:
                axes[ii].annotate(
                    "Not enough data found",
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    horizontalalignment="center",
                )

        if data_found:
            axes[0].set_ylabel("Groundwater level (m.a.s.l.)")
            axes[1].sharex(axes[0])
            axes[1].sharey(axes[0])
            axes[2].sharex(axes[0])
            axes[2].sharey(axes[0])
            axes[0].annotate(
                data_name[0],
                (0.01, 0.99),
                xycoords="axes fraction",
                verticalalignment="top",
            )
            axes[1].annotate(
                data_name[1],
                (0.01, 0.99),
                xycoords="axes fraction",
                verticalalignment="top",
            )
            axes[2].annotate(
                data_name[2],
                (0.01, 0.99),
                xycoords="axes fraction",
                verticalalignment="top",
            )
            u4ax.add_basemap(ax=axes[-1])
            u4pltfmt.map_style(axes[-1], crs=wells_gdf.crs)
            fig.suptitle(k)
            fig.tight_layout()
            fig.savefig(os.path.join(out_path, f"{k}_psi_vs_gwlvl.png"))
        plt.close(fig)


def load_data(
    wells_gdf: gp.GeoDataFrame, fpath: os.PathLike, table: str, k: str
):
    region = wells_gdf["geometry"][wells_gdf["name"] == k].buffer(500)
    well_psi = u4gpkg.load_gpkg_data_region_ogr(
        region, fpath, table, crs=region.crs
    )
    if len(well_psi) > 10:
        return u4conv.reformat_gdf_to_dict(well_psi), well_psi
    else:
        return [], []


def export_to_csv(data: dict, fpath: os.PathLike, gdf: dict):
    header = ["Time", "GW (m.a.s.l)", "Median PSI (m)"]
    for pid in gdf["ps_id"]:
        header.append(f"PS_{pid}")

    rows = [header]
    for ii in range(len(data["time"])):
        row = []
        row.append(datetime.datetime.strftime(data["time"][ii], "%d.%m.%Y"))
        row.append(f"{data['gw'][ii]:.2f}")
        med = data["psi"][ii]
        if np.isnan(med):
            row.append("")
        else:
            row.append(f"{data['psi'][ii]:.2f}")
        for val in list(data["psi_ts"][:, ii]):
            if np.isnan(val):
                row.append("")
            else:
                row.append(f"{val:.2f}")
        rows.append(row)
    with open(fpath, "wt", encoding="utf8", newline="\n") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
