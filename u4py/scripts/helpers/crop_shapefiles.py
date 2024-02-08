""" Crops the tectonics shapefile"""

import os

import u4py.addons.geology as u4geol
import u4py.io.files as u4files


def main():
    tektonik_path = u4files.get_file_paths(filetypes=(("*.shp", "*.shp"),))
    bld_path = u4files.get_file_paths(filetypes=(("*.shp", "*.shp"),))
    tek_hessen = u4geol.get_tektonik_hessen(tektonik_path, bld_path)

    base_path, file_path = os.path.split(tektonik_path)
    fname, _ = os.path.splitext(file_path)
    output_path = os.path.join(base_path, fname + "_cropped.shp")
    tek_hessen.to_file(output_path)


if __name__ == "__main__":
    main()
