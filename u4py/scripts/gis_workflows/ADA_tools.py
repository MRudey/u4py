"""
Creates Options files and calls the CLI of ADAtools

*Requires a valid installation of ADAtools*
"""

import os
import subprocess
from pathlib import Path

from tqdm import tqdm

import u4py.addons.adatools as u4ada
import u4py.utils.projects as u4proj


def main():
    ada_execpath = r"C:\Program Files\cttc\ADAtools\bin"
    project = u4proj.get_project(
        proj_path=Path(
            r"~\Documents\ArcGIS\U4_projects\ADA_tools.u4project"
        ).expanduser(),
        required=[
            "base_path",
            "places_path",
            "output_path",
            "psi_path",
        ],
        interactive=False,
    )

    # Run ADAfinder
    overwrite_ADA = True

    shp_file_list = [
        os.path.join(project["paths"]["psi_path"], fp)
        for fp in os.listdir(project["paths"]["psi_path"])
        if fp.endswith(".shp") and fp.startswith("input_points_")
    ]
    for shp_file in tqdm(shp_file_list, desc="Running ADAfinder", leave=False):
        shpfname = os.path.splitext(os.path.split(shp_file)[-1])[0].replace(
            "input_points_", ""
        )
        adafinder_log_path = os.path.join(
            project["paths"]["output_path"], f"{shpfname}_adafinder.log"
        )
        adafinder_exe_path = os.path.join(ada_execpath, "adafinder_cmd.exe")
        if not os.path.exists(adafinder_log_path) or overwrite_ADA:
            adafinder_cfg_path = u4ada.create_adafinder_config(
                shp_file, project
            )
            with open(adafinder_log_path, "wt") as adafinder_logfile:
                subprocess.run(
                    [adafinder_exe_path, adafinder_cfg_path],
                    stdout=adafinder_logfile,
                    stderr=subprocess.STDOUT,
                )

    # Merge results
    merged_path = os.path.join(
        project["paths"]["output_path"], "merged_ada_results"
    )
    os.makedirs(merged_path, exist_ok=True)
    u4ada.merge_shps(project, "adas.shp", "vertikal")
    u4ada.merge_shps(project, "adas.shp", "Ost_West")


if __name__ == "__main__":
    main()
