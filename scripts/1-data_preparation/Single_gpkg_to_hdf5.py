"""
Converts a gpkg file containing points with timeseries into chunked hdf5 files.

#Parallelized
#SLURM
"""


import logging
import os
import sys

import u4py.utils.cmd_args as u4cmds
import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
import u4py.utils.sql as u4sql

logging.basicConfig(
    format="[%(levelname)s] %(funcName)s: %(message)s",
    stream=sys.stdout,
    level=u4config.log_level,
)


def main():
    # Enable commandline arguments
    u4cmds.load(module_descript=__doc__)

    # Get the file list
    if u4config.in_path:
        file_list = [
            u4config.in_path,
        ]
    else:
        file_list = u4files.get_file_paths(filetypes=(("*.gpkg", "*.gpkg"),))

    # START PROCESSING
    if file_list:
        logging.info("Processing list:")
        for ii, file_name in enumerate(file_list):
            logging.info(f" {ii:03g}: {file_name}")
        chunk_file_list(file_list, chunksize=100)
    else:
        logging.info("File List is empty. Evaluation stopped.")
        return


def chunk_file_list(file_list, chunksize):
    for file_path in file_list:  # tqdm(file_list, desc="Converting files"):
        # Get paths
        base_path, fname_ext = os.path.split(file_path)
        base_path, _ = os.path.split(base_path)
        fname, _ = os.path.splitext(fname_ext)
        export_path = os.path.join(
            base_path, f"Converted_gpkg_chunks-{chunksize}m"
        )
        os.makedirs(export_path, exist_ok=True)

        tables = u4sql.get_table_names(file_path)
        for (
            table
        ) in tables:  # tqdm(tables, desc="Reading from tables", leave=False):
            process_table(table, file_path, export_path, fname, chunksize)


def process_table(
    table: str,
    file_path: os.PathLike,
    export_path: os.PathLike,
    fname: str,
    chunksize: int,
):
    logging.info(f"Processing {table}.")
    data = u4sql.table_to_dict(file_path, table)
    export_path = os.path.join(export_path, "Insar_chunks")
    if data:
        if "l2a" in file_path or "L2A" in file_path:
            export_path = os.path.join(export_path, "L2A")
        elif "l2b" in file_path or "L2B" in file_path:
            export_path = os.path.join(export_path, "L2B")
        elif "l3" in file_path or "L3" in file_path:
            export_path = os.path.join(export_path, "L3")

        if "asce" in file_path:
            chunked_path = os.path.join(export_path, "ASCE")
        elif "desc" in file_path:
            chunked_path = os.path.join(export_path, "DESC")
        elif ("Ost_West" in file_path) or (table == "Ost_West"):
            chunked_path = os.path.join(export_path, "BBD_EW")
        elif ("Vertikal" in file_path) or (table == "vertikal"):
            chunked_path = os.path.join(export_path, "BBD_Vert")

        u4convert.chunk_data_numba(
            data,
            chunked_path,
            chunksize=chunksize,
            min_values=3,
            compress=True,
        )


if __name__ == "__main__":
    main()
