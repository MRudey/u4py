"""
Converts a gpkg file containing points with timeseries into chunked hdf5 files.

#Parallelized
#SLURM
"""


import os
import pickle

import u4py.analysis.processing as u4proc
import u4py.utils.files as u4files
import u4py.utils.sql as u4sql


def main():
    file_path = "/mnt/Raid/Umwelt4/Daten/hessen_l3_clipped.gpkg"

    # START PROCESSING
    output_folder = u4files.multi_split(file_path, 2)
    output_file = os.path.join(output_folder, "extracted_hessen_l3_data.pkl")
    if os.path.exists(output_file):
        with open(output_file, "rb") as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = u4sql.load_tables(file_path)
        with open(output_file, "wb") as pkl_file:
            pickle.dump(data, pkl_file)

    extracts = u4proc.get_extracts(data)
    results = u4proc.batch_mapping(
        extracts, u4proc.inversion_map_worker, "Inverting Extracts"
    )
    # results = [u4proc.inversion_map_worker(ext) for ext in tqdm(extracts)]

    with open(
        os.path.join(output_folder, "all_inversion_results.pkl"),
        "wb",
    ) as pkl_file:
        pickle.dump(results, pkl_file)


if __name__ == "__main__":
    main()
