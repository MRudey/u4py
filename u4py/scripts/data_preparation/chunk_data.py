import os

import numpy as np
from tqdm import tqdm

import u4py.utils.convert as u4convert
import u4py.utils.files as u4files


def main():
    folder_path = u4files.get_folder_paths()
    base_folder, _ = os.path.split(folder_path)
    file_list = []
    # file_list = [
    #     os.path.join(folder_path, f)
    #     for f in os.listdir(folder_path)
    #     if f.endswith(".h5") and "Zeitreihe" in f
    # ]
    file_list.append(os.path.join(folder_path, "BBD_2021_PSI_Ost_West.h5"))
    file_list.append(os.path.join(folder_path, "BBD_2021_PSI_Vertikal.h5"))

    no_time = []
    for file_path in tqdm(file_list, desc="Files", leave=False):
        data = u4files.load_hdf5(file_path)
        if "ASCE" in file_path:
            chunked_path = os.path.join(base_folder, "Insar_chunks", "ASCE")
        elif "DESC" in file_path:
            chunked_path = os.path.join(base_folder, "Insar_chunks", "DESC")
        elif "Ost_West" in file_path:
            chunked_path = os.path.join(base_folder, "Insar_chunks", "BBD_EW")
        elif "Vertikal" in file_path:
            chunked_path = os.path.join(
                base_folder, "Insar_chunks", "BBD_Vert"
            )
        not_chunked = u4convert.chunk_data(
            data, chunked_path, chunksize=250, min_values=3, compress=False
        )
        if not_chunked:
            no_time.append(not_chunked)


if __name__ == "__main__":
    main()
