"""
Extracts all features from a list of geodatabases and stores them in
individual folders for each GDB.
"""
from pathlib import Path

from tqdm import tqdm

import u4py.utils.files as u4files


def main():
    file_list = [
        r"~\Documents\ArcGIS\Places\erdfaelle.gdb",
        r"~\Documents\ArcGIS\Places\steinschlaege.gdb",
        r"~\Documents\ArcGIS\Places\setzungen.gdb",
        r"~\Documents\ArcGIS\Places\karstgefaehrdung.gdb",
        r"~\Documents\ArcGIS\Places\rutschungen.gdb",
    ]

    for file_path in tqdm(file_list, desc="Processing GDB", leave=False):
        u4files.gdb_to_shp(Path(file_path).expanduser())


if __name__ == "__main__":
    main()
