""" Converts a gpkg file containing points with timeseries into hdf5 """


import os
import sys

from tqdm import tqdm

import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def main():
    if len(sys.argv) == 1:
        file_list = u4files.get_file_paths(filetypes=(("*.gpkg", "*.gpkg"),))
    else:
        file_list = [arg for arg in sys.argv[1:]]
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
        if "asce" in file_path:
            chunked_path = os.path.join(export_path, "Insar_chunks", "ASCE")
        elif "desc" in file_path:
            chunked_path = os.path.join(export_path, "Insar_chunks", "DESC")
        elif "Ost_West" in file_path:
            chunked_path = os.path.join(export_path, "Insar_chunks", "BBD_EW")
        elif "Vertikal" in file_path:
            chunked_path = os.path.join(
                export_path, "Insar_chunks", "BBD_Vert"
            )
        u4convert.chunk_data(
            data, chunked_path, chunksize=250, min_values=3, compress=False
        )


if __name__ == "__main__":
    main()
