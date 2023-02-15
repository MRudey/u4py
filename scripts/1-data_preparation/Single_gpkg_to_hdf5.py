""" Converts a gpkg file containing points with timeseries into hdf5 """


import os

from tqdm import tqdm

import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def main():
    file_list = u4files.get_file_paths(filetypes=(("*.gpkg", "*.gpkg"),))
    if not file_list:
        return

    for file_path in tqdm(file_list, desc="Converting files"):
        # Get paths
        base_path, fname_ext = os.path.split(file_path)
        base_path, _ = os.path.split(base_path)
        fname, _ = os.path.splitext(fname_ext)
        export_path = os.path.join(base_path, "Converted_gpkg")
        os.makedirs(export_path, exist_ok=True)

        tables = u4convert.get_table_names(file_path)
        for table in tqdm(tables, desc="Reading from tables", leave=False):
            process_table(table, file_path, export_path, fname)


def process_table(
    table: str, file_path: os.PathLike, export_path: os.PathLike, fname: str
):
    h5path = os.path.join(export_path, f"{table}_{fname}.h5")
    data = u4convert.table_to_dict(file_path, table)
    if data:
        u4convert.dict_to_hdf5(h5path, data)


if __name__ == "__main__":
    main()
