"""
Extracts data from all datasets for use with ADAtools by Navarro et al. 2020
(https://doi.org/10.3390/ijgi9100584)
"""

from pathlib import Path

import u4py.addons.adatools as u4ada
import u4py.utils.config as u4config
import u4py.utils.projects as u4proj

u4config.start_logger()


def main():
    project = u4proj.get_project(
        proj_path=Path(
            "~/Documents/umwelt4/Extract_Data_for_ADAtools.u4project"
        ).expanduser(),
        required=["base_path", "psi_path", "output_path"],
        interactive=False,
    )

    # First test:
    # create_test_region(project)

    # Full conversion
    u4ada.convert_gpkg_to_shp("hessen_l3_clipped.gpkg", project)


if __name__ == "__main__":
    main()
