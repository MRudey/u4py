""" Converts a gpkg file containing points with timeseries into hdf5 """


import argparse
import logging
import os
import sys

import u4py.utils.config as u4config
import u4py.utils.convert as u4convert
import u4py.utils.files as u4files
import u4py.utils.sql as u4sql

logging.basicConfig(
    format="[%(levelname)s] %(funcName)s: %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)


def main():
    # Get commandline arguments
    parser = setup_parser()
    args = parser.parse_args()

    # Get the file list
    if args.file_path:
        file_list = [
            args.file_path,
        ]
    else:
        file_list = u4files.get_file_paths(filetypes=(("*.gpkg", "*.gpkg"),))

    # Adapt logger level when set to verbose (-v)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.info(f"Set loglevel to {log_level(logging.getLogger().level)}.")

    # Adjust number of cpus
    u4config.slurm_cpus = args.cpus
    logging.info(f"Using {u4config.slurm_cpus} CPUs.")

    # START PROCESSING
    if file_list:
        logging.info("Processing list:")
        for ii, file_name in enumerate(file_list):
            logging.info(f" {ii}: {file_name}")
        chunk_file_list(file_list)
    else:
        logging.info("File List is empty. Evaluation stopped.")
        return


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Converts gpkg file(s) to a chunked dataset."
    )
    parser.add_argument(
        "-i",
        "--file_path",
        help="Filepath of the gpkg file. You will be asked to provide one when empty.",
        metavar="path_to_file",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--cpus",
        help="Number of CPUs to use for parallel processing (currently only for SQL queries).",
        metavar="number",
        default=os.cpu_count(),
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Sets the logging level to DEBUG",
        action="store_true",
    )
    return parser


def chunk_file_list(file_list):
    for file_path in file_list:  # tqdm(file_list, desc="Converting files"):
        # Get paths
        base_path, fname_ext = os.path.split(file_path)
        base_path, _ = os.path.split(base_path)
        fname, _ = os.path.splitext(fname_ext)
        export_path = os.path.join(base_path, "Converted_gpkg")
        os.makedirs(export_path, exist_ok=True)

        tables = u4sql.get_table_names(file_path)
        for (
            table
        ) in tables:  # tqdm(tables, desc="Reading from tables", leave=False):
            process_table(table, file_path, export_path, fname)


def process_table(
    table: str, file_path: os.PathLike, export_path: os.PathLike, fname: str
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
        elif "Ost_West" in file_path:
            chunked_path = os.path.join(export_path, "BBD_EW")
        elif "Vertikal" in file_path:
            chunked_path = os.path.join(export_path, "BBD_Vert")

        u4convert.chunk_data_numba(
            data, chunked_path, chunksize=250, min_values=3, compress=True
        )


def log_level(num: int) -> str:
    """Converts loglevel number into string"""
    if num == 50:
        return "CRITICAL"
    elif num == 40:
        return "ERROR"
    elif num == 30:
        return "WARNING"
    elif num == 20:
        return "INFO"
    elif num == 10:
        return "DEBUG"
    elif num == 0:
        return "NOTSET"


if __name__ == "__main__":
    main()
